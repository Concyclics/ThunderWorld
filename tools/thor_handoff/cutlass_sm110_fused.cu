// Thor NVFP4: dense and selected repair terms share the CUTLASS TMA/UMMA
// mainloop, Tensor Memory accumulator and a single BF16 epilogue.
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "sm110_repair_collective.cuh"

using Tile=cute::Shape<cute::_128,cute::_128,cute::_128>;
using Cluster=cute::Shape<cute::_1,cute::_1,cute::_1>;
using FP4=cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using FusedEpilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100,cutlass::arch::OpClassBlockScaledTensorOp,
    Tile,Cluster,cutlass::epilogue::collective::EpilogueTileAuto,
    float,float,void,cutlass::layout::RowMajor,4,
    cutlass::bfloat16_t,cutlass::layout::RowMajor,8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using FusedOriginal=typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,cutlass::arch::OpClassBlockScaledTensorOp,
    FP4,cutlass::layout::RowMajor,32,FP4,cutlass::layout::ColumnMajor,32,
    float,Tile,Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename FusedEpilogue::SharedStorage)>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
template<bool Repair> struct Configuration {
  using Mainloop=cute::conditional_t<Repair,ThorRepairCollective<FusedOriginal>,FusedOriginal>;
  using Kernel=cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int,int>,Mainloop,FusedEpilogue,void>;
  using Operation=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
using Dense=Configuration<false>::Operation;
using Fused=Configuration<true>::Operation;

template<class Op> typename Op::Arguments arguments(int m,int n,int k,
    void const* a,void const* b,void const* sa,void const* sb,void* out,float const* alpha) {
  using Kernel=typename Op::GemmKernel;
  typename Op::Arguments args{};
  args.mode=cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape=cute::make_shape(m,n,k,1);
  args.mainloop.ptr_A=static_cast<cutlass::float_e2m1_t const*>(a);
  args.mainloop.ptr_B=static_cast<cutlass::float_e2m1_t const*>(b);
  args.mainloop.ptr_SFA=static_cast<cutlass::float_ue4m3_t const*>(sa);
  args.mainloop.ptr_SFB=static_cast<cutlass::float_ue4m3_t const*>(sb);
  args.mainloop.dA=cutlass::make_cute_packed_stride(typename Kernel::StrideA{},{m,k,1});
  args.mainloop.dB=cutlass::make_cute_packed_stride(typename Kernel::StrideB{},{n,k,1});
  using Scales=typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  args.mainloop.layout_SFA=Scales::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB=Scales::tile_atom_to_shape_SFB(args.problem_shape);
  args.epilogue.ptr_D=static_cast<typename Op::ElementD*>(out);
  args.epilogue.dD=cutlass::make_cute_packed_stride(typename Kernel::StrideD{},{m,n,1});
  args.epilogue.dC=args.epilogue.dD;
  args.epilogue.thread.alpha=1.f;args.epilogue.thread.alpha_ptr=alpha;
  args.epilogue.thread.beta=0.f;
  return args;
}
extern "C" size_t tw_workspace(int m,int n,int k) {
  auto a=Dense::get_workspace_size(arguments<Dense>(m,n,k,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr));
  auto b=Fused::get_workspace_size(arguments<Fused>(m,n,k,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr));
  return a>b?a:b;
}
template<class Op> int launch(typename Op::Arguments args,void* workspace,void* stream) {
  Op op;auto status=op.can_implement(args);
  if(status!=cutlass::Status::kSuccess)return int(status);
  status=op.initialize(args,workspace,static_cast<cudaStream_t>(stream));
  if(status!=cutlass::Status::kSuccess)return int(status);
  return int(op.run(static_cast<cudaStream_t>(stream)));
}
extern "C" int tw_fused(int m,int n,int k,void const* a,void const* b,
    void const* sa,void const* sb,void* out,float const* alpha,void* workspace,void* stream,ThorRepairIO const* repair) {
  if(!repair->na && !repair->hasw)
    return launch<Dense>(arguments<Dense>(m,n,k,a,b,sa,sb,out,alpha),workspace,stream);
  auto args=arguments<Fused>(m,n,k,a,b,sa,sb,out,alpha);
  args.mainloop.repair=*repair;
  return launch<Fused>(args,workspace,stream);
}
extern "C" char const* tw_status(int status){return cutlass::cutlassGetStatusString(static_cast<cutlass::Status>(status));}
