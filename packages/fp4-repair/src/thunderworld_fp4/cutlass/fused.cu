// Experimental SM120 NVFP4 dense + sparse repair GEMM. CUTLASS headers retain their own license.
// All terms share FP32 registers, then one BF16 epilogue. Empty repair is specialized.
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"

#include "repair_collective.cuh"

#ifndef TW_TILE_N
#define TW_TILE_N 128
#endif
static_assert(TW_TILE_N==64 || TW_TILE_N==128);
using Tile = cute::Shape<cute::_128, cute::Int<TW_TILE_N>, cute::_128>;
using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
using FP4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
template <typename Output,bool Repair=true> struct Configuration {
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, void, cutlass::layout::RowMajor, 4,
    Output, cutlass::layout::RowMajor, 128/cutlass::sizeof_bits<Output>::value,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using OriginalMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    FP4, cutlass::layout::RowMajor, 32,
    FP4, cutlass::layout::ColumnMajor, 32, float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)+(Repair?20*1024:0)>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Mainloop = cute::conditional_t<Repair,RepairCollective<OriginalMainloop>,OriginalMainloop>;
using Kernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>, Mainloop, Epilogue, void>;
using Operation = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
using Operation = Configuration<cutlass::bfloat16_t,false>::Operation;
using DenseOperation = Configuration<cutlass::bfloat16_t>::Operation;

template <typename Op> static typename Op::Arguments arguments(int m, int n, int k, void const* a,
    void const* b, void const* sa, void const* sb, void* output, float const* alpha=nullptr) {
  using Kernel = typename Op::GemmKernel;
  typename Op::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = cute::make_shape(m,n,k,1);
  args.mainloop.ptr_A = static_cast<cutlass::float_e2m1_t const*>(a);
  args.mainloop.ptr_B = static_cast<cutlass::float_e2m1_t const*>(b);
  args.mainloop.ptr_SFA = static_cast<cutlass::float_ue4m3_t const*>(sa);
  args.mainloop.ptr_SFB = static_cast<cutlass::float_ue4m3_t const*>(sb);
  args.mainloop.dA = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {m,k,1});
  args.mainloop.dB = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n,k,1});
  using ScaleLayout = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  args.mainloop.layout_SFA = ScaleLayout::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB = ScaleLayout::tile_atom_to_shape_SFB(args.problem_shape);
  args.epilogue.ptr_D = static_cast<typename Op::ElementD*>(output);
  args.epilogue.dD = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {m,n,1});
  args.epilogue.dC = args.epilogue.dD;
  args.epilogue.thread.alpha = 1.f;
  args.epilogue.thread.alpha_ptr = alpha;
  args.epilogue.thread.beta = 0.f;
  return args;
}

extern "C" size_t tw_workspace(int m,int n,int k) {
  auto fp32 = Operation::get_workspace_size(arguments<Operation>(m,n,k,nullptr,nullptr,nullptr,nullptr,nullptr));
  auto bf16 = DenseOperation::get_workspace_size(arguments<DenseOperation>(m,n,k,nullptr,nullptr,nullptr,nullptr,nullptr));
  return fp32 > bf16 ? fp32 : bf16;
}
template <typename Op> int launch(typename Op::Arguments args,void* workspace,void* stream) {
  Op op;
  auto status = op.can_implement(args);
  if (status != cutlass::Status::kSuccess) return int(status);
  status = op.initialize(args,workspace,static_cast<cudaStream_t>(stream));
  if (status != cutlass::Status::kSuccess) return int(status);
  return int(op.run(static_cast<cudaStream_t>(stream)));
}
extern "C" int tw_fused(int m,int n,int k,void const* a,void const* b,
    void const* sa,void const* sb,void* output,float const* alpha,void* workspace,void* stream,RepairIO const* repair) {
  if (!repair->na && !repair->hasw)
    return launch<Operation>(arguments<Operation>(m,n,k,a,b,sa,sb,output,alpha),workspace,stream);
  auto args=arguments<DenseOperation>(m,n,k,a,b,sa,sb,output,alpha);
  args.mainloop.repair=*repair;
  return launch<DenseOperation>(args,workspace,stream);
}
extern "C" char const* tw_status(int status) {
  return cutlass::cutlassGetStatusString(static_cast<cutlass::Status>(status));
}
