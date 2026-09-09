"""Source-server-only export. Receiver needs only the produced tensors + Git repo."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'packages/fp4-repair/src'))
from thunderworld_fp4.linear import _pack

class Metadata:pass

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def tensor_digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def decode(c,s):
    ids=torch.stack((c&15,c>>4),-1).flatten(-2).long()
    lut=torch.tensor([0,.5,1,1.5,2,3,4,6],device=c.device)
    return lut[ids&7]*torch.where(ids>=8,-1,1)*s.float().repeat_interleave(16,-1)

def quant(x,d):
    c,s,g,_=_pack(x,d,True,True,x)
    return decode(c,s)[:,:x.shape[0]]*g,c,s,g

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--layers',type=int,nargs='+',default=[0]);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','3'):raise RuntimeError('Source host: explicitly use physical GPU 0 or 3')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if a.output.exists():raise FileExistsError('Use a new output directory to preserve exported data')
    a.output.mkdir(parents=True)
    cp=ROOT/'artifacts/external_fp4/Reza2kn_Cosmos3_Nano_NVFP4_AWQ/transformer/modelopt_quantized.pt'
    with torch.serialization.safe_globals([(Metadata,'modelopt.torch.quantization.qtensor.nvfp4_tensor.NVFP4QTensor')]):
        state=torch.load(cp,weights_only=True,map_location='cpu',mmap=True)['model_state_dict']
    source_files={'calibration':'cosmos3_nano_dev_activations_true_steps_cond_gpu0.pt','evaluation':'cosmos3_nano_pai_robot026_dev_activations_true_steps_cond_gpu0.pt'}
    caps={k:torch.load(ROOT/'results/f4'/v,weights_only=True,mmap=True,map_location='cpu')['panels'] for k,v in source_files.items()}
    index=ROOT/'artifacts/huggingface/models/Cosmos3-Nano-411f42a/transformer/diffusion_pytorch_model.safetensors.index.json';idx=json.loads(index.read_text())['weight_map']
    manifest={'schema_version':1,'complete':False,'data_only':True,'model_repo':'nvidia/Cosmos3-Nano','model_files':['transformer/diffusion_pytorch_model.safetensors.index.json'],'base_git_commit':'9711abb09619a6e5e40be9ba0fb71b1fd63a145b','torch':torch.__version__,'cuda':torch.version.cuda,'export_gpu':torch.cuda.get_device_name(),'files':{},'ffn':[],'attention':[],
        'weight_basis':'RTN W4 of BF16-rounded local BF16 W / community AWQ prescale, not external packed AWQ weights',
        'bf16_revision':'411f42a8fdfb8c5b2583cb8786e0938f49796eaa','awq_prescale_revision':'95b7d2328f8ff8667f2f9c55b03d1af122005c16','activation_sources':source_files,
        'scope':'Layer-0 FFN and layer-12 attention development panels by default. Repeating rows for timing creates no new quality evidence.'}
    def save(name,payload):
        path=a.output/name;torch.save(payload,path);manifest['files'][name]={'bytes':path.stat().st_size,'sha256':digest(path)}
        print(name,path.stat().st_size,flush=True)
    for layer in a.layers:
        for kind in ('gate','up','down'):
            name=f'layers.{layer}.mlp_moe_gen.{kind}_proj'
            with safe_open(index.parent/idx[name+'.weight'],framework='pt',device='cpu') as sf:w=sf.get_tensor(name+'.weight').cuda()
            d=state[name+'.input_quantizer._pre_quant_scale'].cuda().float();wt=(w.float()/d).bfloat16().contiguous();wp,wc,ws,wg=quant(wt,torch.ones_like(d));w0=wp[0]
            cal=torch.cat([x[:128] for x in caps['calibration'][name]]).cuda();ap,*_=quant(cal,d);actual=(cal.float()*d).bfloat16().float()
            a_score=(actual-ap[0]).square().mean(0)*w0.square().sum(0);wscore=(wt.float()-w0).square()*actual.square().mean(0)
            n,k=w.shape
            payload={'module':name,'weight_shape':[n,k],'weight_shard':'transformer/'+idx[name+'.weight'],'expected_tensor_sha256':{'weight_bf16':tensor_digest(w),'codes':tensor_digest(wc),'scales':tensor_digest(ws),'global_scale':tensor_digest(wg.reshape(1))},'prescale':d.cpu(),'activation_score':a_score.cpu(),
                'weight_block_scores':{str(b):wscore.reshape(n//128,128,k//b,b).sum((1,3)).cpu() for b in (64,128,256)},
                'eval_inputs':[x[:128].clone().contiguous() for x in caps['evaluation'][name]],'steps':[0,17,34]}
            shard='transformer/'+idx[name+'.weight']
            if shard not in manifest['model_files']:manifest['model_files'].append(shard)
            file=f'ffn_l{layer}_{kind}.pt';save(file,payload);manifest['ffn'].append({'file':file,'module':name,'N':n,'K':k,'real_rows':128})
            del payload,w,wt,wp,wc,ws,w0,cal,ap,actual,wscore;torch.cuda.empty_cache()
    for label,prefix in [('av','cosmos3_nano_dev'),('robot','cosmos3_nano_pai_robot026_dev')]:
        source=ROOT/'results/f4'/f'{prefix}_attention_core_all_layers_true_steps_cond_gpu0.pt'
        cap=torch.load(source,weights_only=True,mmap=True,map_location='cpu')
        for i,row in enumerate(cap['records']['layers.12.self_attn']):
            # clone views so torch.save cannot retain the complete capture storage.
            q=row['q_gen'].unsqueeze(0).clone().contiguous();k=torch.cat([row['k_und'],row['k_gen']]).unsqueeze(0).contiguous();v=torch.cat([row['v_und'],row['v_gen']]).unsqueeze(0).contiguous()
            file=f'attention_{label}_step{[0,17,34][i]}.pt';save(file,{'q':q,'k':k,'v':v,'layer':12,'step':[0,17,34][i],'call_index':row['call_index']})
            manifest['attention'].append({'file':file,'q_shape':list(q.shape),'kv_shape':list(k.shape),'source':str(source.relative_to(ROOT))})
    atlas=ROOT/'results/f4/cosmos3_nano_head_precision_atlas_gpu0.json'
    target=a.output/'head_atlas.json';target.write_bytes(atlas.read_bytes());manifest['files'][target.name]={'bytes':target.stat().st_size,'sha256':digest(target)}
    manifest['complete']=True;manifest['total_bytes']=sum(v['bytes'] for v in manifest['files'].values())
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Complete:',manifest['total_bytes'],'bytes',flush=True)
if __name__=='__main__':main()
