#pragma once
#include <cstdint>

// Extra pointers are immutable except for the current activation planes/mask.
struct RepairIO {
  uint8_t const *ac, *as, *wc, *ws, *wr, *wrs;
  int const *wi, *off, *wloc;
  uint8_t const *mask;
  int m, mp, n, k, na, hasw, cross, shared, dense;
};

template <class Base> struct RepairCollective : Base {
  using typename Base::MainloopPipeline;
  using typename Base::PipelineState;
  using RA=decltype(typename Base::SmemLayoutA{}(cute::_,cute::_,cute::Int<0>{}));
  using RB=decltype(typename Base::SmemLayoutB{}(cute::_,cute::_,cute::Int<0>{}));
  using RSA=decltype(typename Base::SmemLayoutSFA{}(cute::_,cute::_,cute::Int<0>{}));
  using RSB=decltype(typename Base::SmemLayoutSFB{}(cute::_,cute::_,cute::Int<0>{}));
  struct TensorStorage : Base::TensorStorage {
    alignas(1024) cute::ArrayEngine<typename Base::SmemAllocTypeA,cute::cosize_v<RA>> ra;
    alignas(1024) cute::ArrayEngine<typename Base::SmemAllocTypeB,cute::cosize_v<RB>> rb;
    alignas(16) cute::ArrayEngine<typename Base::ElementSF,cute::cosize_v<RSA>> rsa;
    alignas(16) cute::ArrayEngine<typename Base::ElementSF,cute::cosize_v<RSB>> rsb;
  };
  using typename Base::TiledMma;
  struct Arguments : Base::Arguments { RepairIO repair{}; };
  struct Params : Base::Params { RepairIO repair{}; };
  template <class Shape>
  static Params to_underlying_arguments(Shape const& shape, Arguments const& args, void* workspace) {
    Params p{};
    static_cast<typename Base::Params&>(p)=Base::to_underlying_arguments(shape,args,workspace);
    p.repair=args.repair;
    return p;
  }

  template <class Frg, class Coord>
  CUTLASS_DEVICE void mma(MainloopPipeline pipeline, PipelineState state, Frg& accum,
      int count, int tid, TensorStorage& storage, Params const& params, Coord const& coord) {
    Base::mma(pipeline,state,accum,count,tid,storage,params,coord);
    auto const& p=params.repair;
    if (!p.na && !p.hasw) return;
    using namespace cute;
    constexpr int BM=size<0>(typename Base::TileShape{});
    constexpr int BN=size<1>(typename Base::TileShape{});
    static_assert(BM==128 && (BN==128 || BN==64));
    int mb=int(get<0>(coord)), tile_n=int(get<1>(coord));
    int nb=tile_n*BN/128, within_n=tile_n*BN%128;
    TiledMma mma;
    auto thread=mma.get_thread_slice(tid);
    auto allA=make_tensor(make_smem_ptr(storage.ra.begin()),typename Base::SmemLayoutA{});
    auto sA=allA(_,_,Int<0>{});
    auto allB=make_tensor(make_smem_ptr(storage.rb.begin()),typename Base::SmemLayoutB{});
    auto sB=allB(_,_,Int<0>{});
    auto sSA=make_tensor(make_smem_ptr(storage.rsa.begin()),RSA{});
    auto sSB=make_tensor(make_smem_ptr(storage.rsb.begin()),RSB{});
    auto byteA=recast<uint8_t>(sA);auto byteB=recast<uint8_t>(sB);
    auto a=thread.partition_fragment_A(sA);
    auto b=thread.partition_fragment_B(sB);
    auto sa=Base::partition_fragment_SFA(sSA,thread);
    auto sb=Base::partition_fragment_SFB(sSB,thread);
    auto ca=make_tiled_copy_A(typename Base::SmemCopyAtomA{},mma);
    auto cb=make_tiled_copy_B(typename Base::SmemCopyAtomB{},mma);
    auto shape=tile_shape(mma);
    auto csa=make_tiled_copy_impl(typename Base::SmemCopyAtomSFA{},Base::get_layoutSFA_TV(mma),
        make_shape(size<0>(shape),size<2>(shape)));
    auto csb=make_tiled_copy_impl(typename Base::SmemCopyAtomSFB{},Base::get_layoutSFB_TV(mma),
        make_shape(size<1>(shape),size<2>(shape)));
    auto ta=ca.get_thread_slice(tid);auto tb=cb.get_thread_slice(tid);
    auto tsa=csa.get_thread_slice(tid);auto tsb=csb.get_thread_slice(tid);
    auto fromA=ta.partition_S(as_position_independent_swizzle_tensor(allA))(_,_,_,Int<0>{});
    auto fromB=tb.partition_S(as_position_independent_swizzle_tensor(allB))(_,_,_,Int<0>{});
    auto fromSA=tsa.partition_S(as_position_independent_swizzle_tensor(sSA));
    auto fromSB=tsb.partition_S(as_position_independent_swizzle_tensor(sSB));
    auto toA=ta.retile_D(a);auto toB=tb.retile_D(b);
    auto toSA=tsa.retile_D(sa);auto toSB=tsb.retile_D(sb);
    auto sync=[](){cutlass::arch::NamedBarrier::sync(Base::ThreadCount,
        cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);};
    auto term=[&](int kb,bool residual_a,bool residual_w,int location) {
      auto ap=p.ac+int64_t(residual_a)*p.mp*p.k/2+int64_t(mb)*BM*p.k/2+kb*64;
      auto asp=p.as+int64_t(residual_a)*p.mp*(p.k/16)+int64_t(mb)*BM*(p.k/16)+kb*8;
      auto bp=residual_w?p.wr+int64_t(location)*128*64+within_n*64:p.wc+int64_t(tile_n)*BN*p.k/2+kb*64;
      auto bsp=residual_w?p.wrs+int64_t(location)*128*8+within_n*8:p.ws+int64_t(tile_n)*BN*(p.k/16)+kb*8;
      int strideB=residual_w?64:p.k/2;
      CUTLASS_PRAGMA_UNROLL
      for (int i=tid*16;i<128*64;i+=Base::ThreadCount*16) {
        int row=i/64,col=i%64;
        SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>::copy(
            *reinterpret_cast<uint128_t const*>(ap+int64_t(row)*p.k/2+col),
            *reinterpret_cast<uint128_t*>(&byteA(row,col)),mb*BM+row<p.mp);
        if (row<BN) SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>::copy(
            *reinterpret_cast<uint128_t const*>(bp+int64_t(row)*strideB+col),
            *reinterpret_cast<uint128_t*>(&byteB(row,col)),true);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i=tid*4;i<128*8;i+=Base::ThreadCount*4) {
        int row=i/8,col=i%8;
        SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint32_t>::copy(
            *reinterpret_cast<uint32_t const*>(asp+int64_t(row)*(p.k/16)+col),
            *reinterpret_cast<uint32_t*>(&sSA(row,col*16)),mb*BM+row<p.mp);
        if (row<BN) SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint32_t>::copy(
            *reinterpret_cast<uint32_t const*>(bsp+int64_t(row)*(strideB/8)+col),
            *reinterpret_cast<uint32_t*>(&sSB(row,col*16)),true);
      }
      cp_async_fence();cp_async_wait<0>();sync();
      for_each(make_int_sequence<size<2>(a)>{},[&](auto block){
        copy(ca,fromA(_,_,block),toA(_,_,block));
        copy(cb,fromB(_,_,block),toB(_,_,block));
        fp4_shift_A(typename TiledMma::MMA_Op{},toA(_,_,block));
        fp4_shift_B(typename TiledMma::MMA_Op{},toB(_,_,block));
        copy(fromSA(_,_,block),toSA(_,_,block));
        copy(fromSB(_,_,block),toSB(_,_,block));
        cute::gemm(mma,make_zip_tensor(a(_,_,block),sa(_,_,block)),
            make_zip_tensor(b(_,_,block),sb(_,_,block)),accum);
      });
      sync();
    };
    if (p.na) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (int kb=0;kb<p.k/128;++kb)
        if (p.na==p.k/128 || p.mask[mb*(p.k/128)+kb]) term(kb,true,false,0);
    }
    if (p.hasw) {
      int begin=p.dense?nb*(p.k/128):p.off[nb];
      int end=p.dense?(nb+1)*(p.k/128):p.off[nb+1];
      CUTLASS_PRAGMA_NO_UNROLL
      for (int slot=begin;slot<end;++slot) {
        int kb=p.dense?slot-begin:p.wi[slot];
        int location=p.shared?p.wloc[slot]:slot;
        term(kb,false,true,location);
        if (p.cross && p.na && p.mask[mb*(p.k/128)+kb]) term(kb,true,true,location);
      }
    }
  }
};
