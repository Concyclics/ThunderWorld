"""FP4 repair terms seeded by an unrounded FP32 CUTLASS base product.

All selected A/W/cross terms remain FP4 MMA and finish in one BF16 epilogue.
Base is completed before A repair, unlike the historical interleaved order.
The boundary adds one FP32 MxN buffer; no independent BF16 repair products.
"""
import triton
import triton.language as tl


@triton.jit
def repair_tail(BASE,AC,AS,WC,WS,WG,AG,AM,WR,WRS,WI,OFF,WLOC,Y,
                M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,MP:tl.constexpr,
                NA:tl.constexpr,HASW:tl.constexpr,CROSS:tl.constexpr,
                DENSE_W:tl.constexpr,SHARED_BANK:tl.constexpr):
    mb=tl.program_id(0);nb=tl.program_id(1)
    mi=mb*128+tl.arange(0,128);ni=nb*128+tl.arange(0,128)
    dk=tl.arange(0,64);ds=tl.arange(0,8)
    acc=tl.load(BASE+mi[:,None]*N+ni[None,:],mi[:,None]<M,0.)
    if NA>0:
        for kb in range(K//128):
            selected=True if NA==K//128 else tl.load(AM+mb*(K//128)+kb)
            if selected:
                ai=MP*(K//2)+mi[:,None]*(K//2)+kb*64+dk[None,:]
                asi=MP*(K//16)+mi[:,None]*(K//16)+kb*8+ds[None,:]
                wi=ni[:,None]*(K//2)+kb*64+dk[None,:]
                wsi=ni[:,None]*(K//16)+kb*8+ds[None,:]
                av=tl.load(AC+ai,mi[:,None]<MP,0)
                asc=tl.load(AS+asi,mi[:,None]<MP,0.)
                wv=tl.load(WC+wi);wsc=tl.load(WS+wsi)
                acc=tl.dot_scaled(av,asc,'e2m1',wv.T,wsc,'e2m1',acc)
    if HASW:
        begin=nb*(K//128) if DENSE_W else tl.load(OFF+nb)
        end=(nb+1)*(K//128) if DENSE_W else tl.load(OFF+nb+1)
        for slot in range(begin,end):
            kb=slot-begin if DENSE_W else tl.load(WI+slot)
            storage=tl.load(WLOC+slot) if SHARED_BANK else slot
            ai=mi[:,None]*(K//2)+kb*64+dk[None,:]
            asi=mi[:,None]*(K//16)+kb*8+ds[None,:]
            av=tl.load(AC+ai,mi[:,None]<MP,0);asc=tl.load(AS+asi,mi[:,None]<MP,0.)
            wv=tl.load(WR+storage*128*64+tl.arange(0,128)[:,None]*64+dk[None,:])
            wsc=tl.load(WRS+storage*128*8+tl.arange(0,128)[:,None]*8+ds[None,:])
            acc=tl.dot_scaled(av,asc,'e2m1',wv.T,wsc,'e2m1',acc)
            if CROSS and NA>0:
                if tl.load(AM+mb*(K//128)+kb):
                    av=tl.load(AC+MP*(K//2)+ai,mi[:,None]<MP,0)
                    asc=tl.load(AS+MP*(K//16)+asi,mi[:,None]<MP,0.)
                    acc=tl.dot_scaled(av,asc,'e2m1',wv.T,wsc,'e2m1',acc)
    tl.store(Y+mi[:,None]*N+ni[None,:],acc*(tl.load(WG)*tl.load(AG)),mi[:,None]<M)
