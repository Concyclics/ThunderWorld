#pragma once
#include <cstdint>

// Host-owned descriptors/weight metadata; activation planes and masks are current
// Graph inputs. All selected terms use the original M128/N128/K128 lattice.
struct ThorRepairIO {
  uint8_t const *ac, *as, *wc, *ws, *wr, *wrs;
  int const *wi, *off, *wloc;
  uint8_t const *mask;
  int m, mp, n, k, na, hasw, cross, shared, dense, bank_blocks;
};

template<class Base> struct ThorRepairCollective : Base {
  using typename Base::MainloopPipeline;
  using typename Base::MainloopPipelineState;
  using typename Base::TensorStorage;
  using typename Base::TiledMma;
  using typename Base::TileShape;
  using ClusterShape=typename Base::DispatchPolicy::ClusterShape;
  using BaseParams=typename Base::Params;
  struct Arguments : Base::Arguments { ThorRepairIO repair{}; };
  struct Params : BaseParams {
    ThorRepairIO repair{};
    typename BaseParams::TMA_A tma_ar;
    typename BaseParams::TMA_B tma_wr;
  };
  template<class Shape>
  static Params to_underlying_arguments(Shape const& shape,Arguments const& args,void* workspace,
      cutlass::KernelHardwareInfo const& hw={}) {
    using namespace cute;
    Params out{};
    static_cast<BaseParams&>(out)=Base::to_underlying_arguments(shape,args,workspace,hw);
    out.repair=args.repair;
    auto const& p=args.repair;
    auto cluster=tiled_divide(make_layout(ClusterShape{}),make_tile(typename TiledMma::AtomThrID{}));
    auto ap=p.ac+int64_t(p.na>0)*p.mp*p.k/2;
    auto bp=p.hasw?p.wr:p.wc;
    int rows=p.hasw?p.bank_blocks*128:128;
    auto ar=make_tensor(recast_ptr<typename Base::TmaInternalElementA>(ap),
        make_layout(make_shape(p.m,p.k,1),args.dA));
    typename Base::StrideB bd=cutlass::make_cute_packed_stride(typename Base::StrideB{}, {rows,128,1});
    auto wr=make_tensor(recast_ptr<typename Base::TmaInternalElementB>(bp),
        make_layout(make_shape(rows,128,1),bd));
    out.tma_ar=make_tma_atom_A_sm100<typename Base::TmaInternalElementA>(typename Base::GmemTiledCopyA{},
        ar,typename Base::SmemLayoutA{}(_,_,_,Int<0>{}),TileShape{},TiledMma{},cluster);
    out.tma_wr=make_tma_atom_B_sm100<typename Base::TmaInternalElementB>(typename Base::GmemTiledCopyB{},
        wr,typename Base::SmemLayoutB{}(_,_,_,Int<0>{}),TileShape{},TiledMma{},cluster);
    return out;
  }

  CUTLASS_DEVICE ThorRepairCollective(Params const& p,ClusterShape cluster,uint32_t rank)
      : Base(p,cluster,rank),p_(&p) {
    static_assert(cute::size(ClusterShape{})==1 && cute::size(typename TiledMma::AtomThrID{})==1);
    static_assert(cute::size<0>(TileShape{})==128 && cute::size<1>(TileShape{})==128 && cute::size<2>(TileShape{})==128);
  }
  CUTLASS_DEVICE void prefetch_tma_descriptors() {
    Base::prefetch_tma_descriptors();
    cute::prefetch_tma_descriptor(p_->tma_ar.get_tma_descriptor());
    cute::prefetch_tma_descriptor(p_->tma_wr.get_tma_descriptor());
  }
  template<class B,class A,class W> struct LoadInputs {
    decltype(B::k_tiles) k_tiles;
    B base;A ar;W wr;
  };
  template<class Shape>
  CUTLASS_DEVICE auto load_init(Shape const& shape,TensorStorage& storage) {
    using namespace cute;
    storage_=&storage;
    auto base=Base::load_init(shape,storage);
    auto const& p=p_->repair;
    auto ar=p_->tma_ar.get_tma_tensor(make_shape(p.m,p.k,1));
    auto wr=p_->tma_wr.get_tma_tensor(make_shape(p.hasw?p.bank_blocks*128:128,128,1));
    auto ga=local_tile(ar,TileShape{},make_coord(_,_,_),Step<_1,Underscore,_1>{});
    auto gw=local_tile(wr,TileShape{},make_coord(_,_,_),Step<Underscore,_1,_1>{});
    auto cta=TiledMma{}.get_slice(0);
    auto pa=cta.partition_A(ga);auto pw=cta.partition_B(gw);
    auto sa=make_tensor(make_smem_ptr(storage.smem_A.begin()),typename Base::SmemLayoutA{});
    auto sb=make_tensor(make_smem_ptr(storage.smem_B.begin()),typename Base::SmemLayoutB{});
    auto [ta,unused_a]=tma_partition(p_->tma_ar,Int<0>{},make_layout(Int<1>{}),group_modes<0,3>(sa),group_modes<0,3>(pa));
    auto [tw,unused_w]=tma_partition(p_->tma_wr,Int<0>{},make_layout(Int<1>{}),group_modes<0,3>(sb),group_modes<0,3>(pw));
    return LoadInputs<decltype(base),decltype(ta),decltype(tw)>{base.k_tiles,base,ta,tw};
  }

  // Repair operands occupy ordinary mainloop stages. TMA moves codes while the
  // producer warp copies the native K16 scales into CUTLASS's shared layout.
  // Transaction completion publishes a stage only after both have completed.
  template<class Inputs,class Coord>
  CUTLASS_DEVICE void repair_load(MainloopPipeline pipe,MainloopPipelineState& state,
      Inputs const& inputs,Coord const& coord,int kb,bool ra,bool rw,int location) {
    using namespace cute;
    auto const& p=p_->repair;auto const& b=inputs.base;
    int mb=int(get<0>(coord)),nb=int(get<1>(coord)),lane=threadIdx.x%32;
    pipe.producer_acquire(state);
    auto* barrier=pipe.producer_get_barrier(state);
    int stage=state.index();
    if (elect_one_sync()) {
      if (ra) copy(p_->tma_ar.with(*barrier,b.mcast_mask_a),inputs.ar(_,mb,kb,0),b.tAsA(_,stage));
      else copy(this->observed_tma_load_a_->with(*barrier,b.mcast_mask_a),b.tAgA_mkl(_,mb,kb,0),b.tAsA(_,stage));
      if (rw) copy(p_->tma_wr.with(*barrier,b.mcast_mask_b),inputs.wr(_,location,0,0),b.tBsB(_,stage));
      else copy(this->observed_tma_load_b_->with(*barrier,b.mcast_mask_b),b.tBgB_nkl(_,nb,kb,0),b.tBsB(_,stage));
      if (!ra) copy(this->observed_tma_load_sfa_->with(*barrier,b.mcast_mask_sfa),b.tAgSFA_mkl(_,mb,kb,0),b.tAsSFA(_,stage));
      if (!rw) copy(this->observed_tma_load_sfb_->with(*barrier,b.mcast_mask_sfb),b.tBgSFB_nkl(_,nb,kb,0),b.tBsSFB(_,stage));
    }
    auto sa=make_tensor(make_smem_ptr(storage_->smem_SFA.begin()),typename Base::SmemLayoutSFA{});
    auto sb=make_tensor(make_smem_ptr(storage_->smem_SFB.begin()),typename Base::SmemLayoutSFB{});
    auto asp=p.as+int64_t(ra)*p.mp*(p.k/16)+int64_t(mb)*128*(p.k/16)+kb*8;
    auto wsp=rw?p.wrs+int64_t(location)*1024:p.ws+int64_t(nb)*128*(p.k/16)+kb*8;
    constexpr int AK=size<0,1>(typename Base::SmemLayoutSFA{});
    constexpr int BK=size<0,1>(typename Base::SmemLayoutSFB{});
    CUTLASS_PRAGMA_UNROLL
    for (int i=lane*4;i<128*8;i+=32*4) {
      int row=i/8,col=i%8;
      auto ca=make_coord(make_coord(row,(col*16)%AK),0,(col*16)/AK,stage);
      auto cb=make_coord(make_coord(row,(col*16)%BK),0,(col*16)/BK,stage);
      if (ra) SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint32_t>::copy(
          *reinterpret_cast<uint32_t const*>(asp+int64_t(row)*(p.k/16)+col),
          *reinterpret_cast<uint32_t*>(&sa(ca)),mb*128+row<p.mp);
      if (rw) SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint32_t>::copy(
          *reinterpret_cast<uint32_t const*>(wsp+int64_t(row)*(rw?8:p.k/16)+col),
          *reinterpret_cast<uint32_t*>(&sb(cb)),true);
    }
    cp_async_fence();cp_async_wait<0>();__syncwarp();
    cutlass::arch::fence_view_async_shared();
    static_assert(Base::SFTransactionBytes==2048);
    if (elect_one_sync()) cutlass::arch::ClusterTransactionBarrier::complete_transaction(barrier,0,1024*(int(ra)+int(rw)),1);
    ++state;
  }

  template<class Inputs,class Coord,class Iterator>
  CUTLASS_DEVICE auto load(MainloopPipeline pipe,MainloopPipelineState state,
      Inputs const& inputs,Coord const& coord,Iterator iter,int count) {
    auto [next,next_iter]=Base::load(pipe,state,inputs.base,coord,iter,count);
    auto const& p=p_->repair;
    // CUTLASS splits each tile load into a prologue and a remainder. Append
    // once, including the small-K case where the remainder has count zero.
    if (count>0 && int(*iter)+count==p.k/128) {
      int mb=int(cute::get<0>(coord)),nb=int(cute::get<1>(coord));
      if (p.na) for (int kb=0;kb<p.k/128;++kb)
        if (p.na==p.k/128 || p.mask[mb*(p.k/128)+kb]) repair_load(pipe,next,inputs,coord,kb,true,false,0);
      if (p.hasw) {
        int begin=p.dense?nb*(p.k/128):p.off[nb];
        int end=p.dense?(nb+1)*(p.k/128):p.off[nb+1];
        for (int slot=begin;slot<end;++slot) {
          int kb=p.dense?slot-begin:p.wi[slot];
          int location=p.shared?p.wloc[slot]:slot;
          repair_load(pipe,next,inputs,coord,kb,false,true,location);
          if (p.cross && p.na && p.mask[mb*(p.k/128)+kb]) repair_load(pipe,next,inputs,coord,kb,true,true,location);
        }
      }
    }
    return cute::make_tuple(next,next_iter);
  }

  template<class Pipelines,class States,class Acc,class Inputs,class Coord>
  CUTLASS_DEVICE auto mma(Pipelines pipes,States states,Acc const& acc,Inputs const& inputs,Coord coord,int count) {
    auto const& p=p_->repair;int mb=int(cute::get<0>(coord)),nb=int(cute::get<1>(coord));
    int extra=0;
    // Read the current support, not its quota: energy-capped selectors can
    // choose fewer blocks. Producer and consumer must count exactly alike.
    if (p.na) for(int kb=0;kb<p.k/128;++kb) extra+=p.na==p.k/128 || p.mask[mb*(p.k/128)+kb];
    if (p.hasw) {
      int begin=p.dense?nb*(p.k/128):p.off[nb];
      int end=p.dense?(nb+1)*(p.k/128):p.off[nb+1];extra+=end-begin;
      if (p.cross && p.na) for(int slot=begin;slot<end;++slot) {
        int kb=p.dense?slot-begin:p.wi[slot];extra+=bool(p.mask[mb*(p.k/128)+kb]);
      }
    }
    return Base::mma(pipes,states,acc,inputs,coord,count+extra);
  }
private:
  Params const* p_;
  TensorStorage* storage_{};
};
