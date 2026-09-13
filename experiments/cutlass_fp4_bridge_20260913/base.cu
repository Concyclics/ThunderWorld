// Experimental SM120 NVFP4 base GEMM. CUTLASS headers retain their own license.
// Repair consumes an unscaled FP32 accumulator; empty repair writes BF16 directly.
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"

using Tile = cute::Shape<cute::_128, cute::_128, cute::_128>;
using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
using FP4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
template <typename Output> struct Configuration {
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, void, cutlass::layout::RowMajor, 4,
    Output, cutlass::layout::RowMajor, 128/cutlass::sizeof_bits<Output>::value,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    FP4, cutlass::layout::RowMajor, 32,
    FP4, cutlass::layout::ColumnMajor, 32, float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>, Mainloop, Epilogue, void>;
using Operation = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
using Operation = Configuration<float>::Operation;
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
extern "C" int tw_base(int m,int n,int k,void const* a,void const* b,
    void const* sa,void const* sb,float* output,void* workspace,void* stream) {
  return launch<Operation>(arguments<Operation>(m,n,k,a,b,sa,sb,output),workspace,stream);
}
extern "C" int tw_dense(int m,int n,int k,void const* a,void const* b,
    void const* sa,void const* sb,void* output,float const* alpha,void* workspace,void* stream) {
  return launch<DenseOperation>(arguments<DenseOperation>(m,n,k,a,b,sa,sb,output,alpha),workspace,stream);
}
extern "C" char const* tw_status(int status) {
  return cutlass::cutlassGetStatusString(static_cast<cutlass::Status>(status));
}
