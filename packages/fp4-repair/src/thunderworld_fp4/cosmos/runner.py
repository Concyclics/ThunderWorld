"""Complete resident Generator requests, with optional diagnostics and no global patches."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import types
import numpy as np
import torch
from .engine import JointEngine
from .diagnostics import Diagnostics
from .tuning import LaunchTable
from .assets import compact, expand, required_weight_fraction


def model_identity(path):
    path=Path(path)
    files=['transformer/config.json','transformer/diffusion_pytorch_model.safetensors.index.json']
    return {name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in files}


class CosmosRunner:
    def __init__(self,model,config,*,tuning=None,tune=False,assets=None,audio=False):
        if config.resource_mode == 'shared-calibration' and tune:
            raise ValueError('Shared calibration prohibits timing-based tuning')
        self.path=Path(model);self.config=config;self.audio=audio
        self.table=LaunchTable(tuning,tune=tune)
        self.assets_path=Path(assets) if assets else None
        self.pipe=None;self.engine=None;self.diagnostics=None;self.timings={}
        self._warmed=set();self.identity=model_identity(self.path)
        if config.budget_plan:
            from .budget import RepairPlan
            plan=RepairPlan.load(config.budget_plan)
            if plan.value.get('model_identity')!=self.identity:raise ValueError('Budget plan/model identity mismatch')
        self.memory={}
        self.asset_identity=None

    @torch.inference_mode()
    def prepare(self,export=None):
        if self.pipe is not None:raise RuntimeError('Runner already prepared')
        start=time.perf_counter()
        import diffusers.utils.import_utils as imports
        imports._xformers_available=False
        from diffusers import Cosmos3OmniPipeline, Cosmos3OmniTransformer
        from diffusers.models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer
        transformer,info=Cosmos3OmniTransformer.from_pretrained(str(self.path/'transformer'),
            torch_dtype=torch.bfloat16,local_files_only=True,output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise RuntimeError(info)
        audio=None
        if self.audio:
            audio=Cosmos3AVAEAudioTokenizer.from_pretrained(str(self.path/'sound_tokenizer'),
                torch_dtype=torch.bfloat16,local_files_only=True)
        self.pipe=Cosmos3OmniPipeline.from_pretrained(str(self.path),transformer=transformer,
            sound_tokenizer=audio,safety_checker=None,enable_safety_checker=False,
            torch_dtype=torch.bfloat16,local_files_only=True)
        self.pipe.transformer.set_attention_backend('_native_flash')
        self.pipe.vae.enable_tiling();self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to('cuda')
        torch.cuda.synchronize();self.timings['load_seconds']=time.perf_counter()-start
        assets=None
        policy=json.loads(Path(self.config.ffn_policy).read_text()) if self.config.ffn_policy else None
        if self.assets_path:
            manifest=json.loads((self.assets_path/'manifest.json').read_text())
            self.asset_identity=hashlib.sha256((self.assets_path/'manifest.json').read_bytes()).hexdigest()
            if not manifest['complete'] or manifest['model_identity']!=self.identity or manifest['precision']!=self.config.precision:
                raise ValueError('Prepared asset/model/precision mismatch')
            assets={}
            for name,entry in manifest['modules'].items():
                path=self.assets_path/entry['file']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Corrupt prepared asset')
                asset=torch.load(path,weights_only=True,map_location='cpu')
                assets[name]=expand(asset,required_weight_fraction(self.config,name,policy)) if self.config.precision=='fp4' else asset
        start=time.perf_counter()
        self.engine=JointEngine(self.pipe.transformer,self.config,self.table,assets)
        torch.cuda.synchronize();self.timings['offline_prepare_seconds']=time.perf_counter()-start
        if export:
            folder=Path(export);folder.mkdir(parents=True,exist_ok=False)
            manifest=dict(complete=False,model_identity=self.identity,precision=self.config.precision,
                          targets=self.engine.targets,modules={})
            for i,(name,op) in enumerate(self.engine.linears.items()):
                asset=op.export()
                if self.config.precision=='fp4':asset=compact(asset,required_weight_fraction(self.config,name,policy))
                path=folder/f'{i:04d}.pt';torch.save(asset,path)
                manifest['modules'][name]=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            manifest['complete']=True
            (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        # There is no diagnostic hook at all in off mode. The lightweight call
        # wrapper is shared by Graph and eager so request semantics stay equal.
        if self.config.debug!='off':
            engine=self.engine
            def instrumented(*args,**kwargs):
                with self.diagnostics.begin():result=engine(*args,**kwargs)
                self.diagnostics.end(result)
                return result
            self.pipe.transformer.forward=instrumented
        # Always expose sound latents, and optionally decode the same latents
        # inside the measured request (not a separate process estimate).
        self._decode_sound=self.pipe.decode_sound
        self.sound_latent=None;self.audio_seconds=None
        def sound(latent):
            self.sound_latent=latent.detach()
            if self._audio_output=='latent':return latent
            return self._decode_sound(latent)
        self.pipe.decode_sound=sound
        if self.config.debug!='off':
            for obj,method,label in [(self.pipe,'_encode_video','encode_video'),
                                     (self.pipe.vae,'decode','decode_video'),
                                     (self.pipe,'decode_sound','decode_sound')]:
                original=getattr(obj,method)
                def measured(*args,_fn=original,_label=label,**kwargs):
                    return self.diagnostics.stage(_label,_fn,*args,**kwargs)
                setattr(obj,method,measured)
        self.memory['after_prepare']=dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
        return self

    def _args(self,case,seed,steps):
        from PIL import Image
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition
        args=dict(prompt=case['prompt'],negative_prompt=case.get('negative_prompt','distorted objects, flickering, blurry, low quality'),
            num_inference_steps=steps,generator=torch.Generator(device='cuda').manual_seed(seed),
            output_type='np',add_resolution_template=False,add_duration_template=False,
            enable_safety_check=False,enable_sound=case['task']=='av')
        image=Image.open(case['image']).convert('RGB') if case.get('image') else None
        if case['task']=='policy':
            if image is None:raise ValueError('Policy requires an image')
            args.update(action=CosmosActionCondition(mode='policy',chunk_size=16,
                domain_name=case.get('domain','bridge_orig_lerobot'),resolution_tier=256,
                image=image,view_point='third_person_view'),fps=5.,guidance_scale=1.)
        else:
            args.update(height=case.get('height',192),width=case.get('width',320),
                        num_frames=case.get('frames',61),fps=24.,guidance_scale=6.)
            if image is not None:args['image']=image
        return args

    def _context(self,case,steps,audio_output):
        from .scheduler import HostScheduleUniPC
        if case['task']=='av' and not self.audio:raise ValueError('Prepare with audio=True for AV')
        if audio_output not in ('latent','waveform'):raise ValueError('Invalid audio_output')
        self._audio_output=audio_output;self.sound_latent=None
        self.pipe.scheduler=HostScheduleUniPC.from_config(self.pipe.scheduler.config,
            flow_shift=10. if case['task']=='policy' else 3.,use_karras_sigmas=False)
        branches=1 if case['task']=='policy' else 2
        self.engine.reset_request(steps,branches,case['task'])
        if self.diagnostics is None or len(self.diagnostics.events)<steps*branches and self.config.debug!='off':
            self.diagnostics=Diagnostics(self.config.debug,steps*branches)
        self.diagnostics.reset()

    @staticmethod
    def case_key(case,steps,audio_output):
        return json.dumps([case,steps,audio_output],sort_keys=True)

    @torch.inference_mode()
    def warmup(self,case,steps=35,audio_output='waveform'):
        if not self.pipe:raise RuntimeError('Call prepare first')
        key=self.case_key(case,steps,audio_output)
        if key in self._warmed:return
        start=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        self._context(case,steps,audio_output)
        if self.config.compute_budget is not None:
            if steps!=35:raise ValueError('Compute plans require exactly 35 denoising steps')
            self.engine.warming=True;self.engine.discovering=True
            try:
                self.pipe(**self._args(case,6099,2))
                self.engine.compile_budget()
            finally:
                self.engine.warming=False;self.engine.discovering=False
            self._context(case,steps,audio_output)
        self.engine.warming=True
        try:self.pipe(**self._args(case,6099,2))
        finally:self.engine.warming=False
        torch.cuda.synchronize()
        self.timings['warmup_seconds']=self.timings.get('warmup_seconds',0)+time.perf_counter()-start
        self._warmed.add(key)
        self.memory['after_warmup']=dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if self.config.strict_tuning and self.table.misses:
            raise RuntimeError(f'Unmatched tuning entries: {len(self.table.misses)}; warmup with --tune')

    @torch.inference_mode()
    def infer(self,case,seed,steps=35,audio_output='waveform'):
        if self.case_key(case,steps,audio_output) not in self._warmed:raise RuntimeError('Warmup this request before infer')
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        self._context(case,steps,audio_output)
        args=self._args(case,seed,steps)
        output=self.pipe(**args)
        signals={'video':np.asarray(output.video)}
        if output.action is not None:signals['action']=output.action[0].float().cpu().numpy()
        if self.sound_latent is not None:signals['sound_latent']=self.sound_latent.float().cpu().numpy()
        if output.sound is not None and audio_output=='waveform':signals['waveform']=output.sound.float().cpu().numpy()
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        if self.engine.call!=steps*self.engine.branches:raise RuntimeError('Incorrect denoising call count')
        if not all(np.isfinite(x).all() for x in signals.values()):raise RuntimeError('Nonfinite output')
        stats=dict(seed=seed,steps=steps,case=case['id'],task=case['task'],wall_seconds=seconds,
            audio_output=audio_output,debug=self.config.debug,execution=self.config.execution,
            performance_eligible=self.config.debug=='off' and self.config.resource_mode=='exclusive',
            resource_mode=self.config.resource_mode,compute_ledger=self.engine.budget_ledger,
            timing_scope='complete_request_v2_including_host_setup',
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            captures=len(self.engine.cache),calls=self.engine.call,
            phase_replays={str(k):v for k,v in self.engine.replays.items()},
            diagnostics=self.diagnostics.finish(),
            sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in signals.items()})
        if self.config.debug=='trace':
            stats['diagnostics']['attention_masks']=[{str(i):m.cpu().tolist() for i,m in row.items()} for row in self.engine.mask_history]
        return signals,stats

    def benchmark(self,case,seeds,steps=35,audio_output='waveform'):
        if self.config.debug!='off':raise ValueError('Formal benchmark requires debug off')
        if not self.config.strict_tuning:raise ValueError('Formal benchmark requires strict_tuning')
        if self.config.resource_mode == 'shared-calibration':
            raise RuntimeError('Shared calibration results are ineligible for benchmark')
        from .cli import idle_guard
        idle_guard()
        self.warmup(case,steps,audio_output)
        self.warm_request(case,steps,audio_output)
        rows=[]
        for seed in seeds:
            idle_guard()
            _,row=self.infer(case,seed,steps,audio_output)
            idle_guard()
            rows.append(row)
        return rows

    def warm_request(self,case,steps=35,audio_output='waveform'):
        """One complete untabulated request after all compilation/capture."""
        _,row=self.infer(case,6099,steps,audio_output)
        self.timings['full_request_warmup_seconds']=row['wall_seconds']

    def set_repair_masks(self,masks):
        """Replace owned per-(layer, CFG branch) masks; caller must warm up again."""
        if self.engine is None:raise RuntimeError('Prepare before installing masks')
        checked={}
        for (layer,branch),mask in masks.items():
            if layer not in range(len(self.pipe.transformer.layers)) or branch not in (0,1):
                raise ValueError('Invalid mask layer/branch')
            if mask.dtype!=torch.bool or not mask.is_cuda or mask.ndim!=3:
                raise ValueError('Expected CUDA bool [head, M128, N128] mask')
            checked[layer,branch]=mask.contiguous().clone()
        self.engine.clear();self.engine.explicit_masks=checked
        self.engine.attention_prepared.clear();self._warmed.clear()

    def receipt(self):
        storages={}
        scratch={}
        for op in self.engine.linears.values():
            if hasattr(op,'ops'):
                for native in op.ops.values():
                    for name in ('wc','ws','wg','wr','wrs','wi','off','d','wloc'):
                        value=getattr(native,name,None)
                        if value is None:continue
                        storage=value.untyped_storage()
                        storages[storage.data_ptr()]=storage.nbytes()
                    for name in ('am','union','ai'):
                        storage=getattr(native,name).untyped_storage()
                        scratch[storage.data_ptr()]=storage.nbytes()
            else:
                for value in op.buffers():
                    storage=value.untyped_storage();storages[storage.data_ptr()]=storage.nbytes()
        model_state=dict(storages)
        for component in self.pipe.components.values():
            if not isinstance(component,torch.nn.Module):continue
            for value in (*component.parameters(),*component.buffers()):
                if value.is_cuda:
                    storage=value.untyped_storage();model_state[storage.data_ptr()]=storage.nbytes()
        return dict(config=self.config.to_dict(),model_identity=self.identity,environment=self.table.identity,
            asset_identity=self.asset_identity,
            ffn_policy_identity=None if self.engine.policy is None else hashlib.sha256(
                json.dumps(self.engine.policy,sort_keys=True).encode()).hexdigest(),
            timings={**self.timings,'graph_capture_seconds':self.engine.graph_capture_seconds},
            targets=self.engine.targets,observed_shapes=list(self.engine.shapes.values()),
            warmup_gqa_union_density={str(key):row['warmup_union_density'] for key,row in self.engine.attention_shapes.items() if 'warmup_union_density' in row},
            packing_density_scope='Warmup samples only; request cost bound uses arbitrary online support unions',
            tuning_missing=sorted(self.table.misses),tuning_hits=sorted(self.table.hits),
            prepared_weight_bytes=sum(storages.values()),support_buffer_bytes=sum(scratch.values()),
            resident_model_state_bytes=sum(model_state.values()),
            memory=self.memory,
            source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in Path(__file__).resolve().parents[1].rglob('*.py')})
