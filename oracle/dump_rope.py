"""Cross-check: C++ RopeSpec::inv_freq (include/laguna_config.h) vs the shipped
transformers rotary implementation, on the REAL config."""
import os,sys,json,torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from tiny_ref import _MD, LagunaConfig
from ref_laguna import yarn_inv_freq
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
import copy
cfg=json.load(open(os.path.join(_MD,'config.json')))
o=LagunaConfig(**{k:v for k,v in cfg.items() if k not in ('architectures','auto_map','quantization_config','model_type','dtype','torch_dtype')})
res={}
for name,key in (("full","full_attention"),("slide","sliding_attention")):
    c2=copy.deepcopy(o); c2.rope_parameters=dict(cfg['rope_parameters'][key])
    c2.partial_rotary_factor=c2.rope_parameters.get('partial_rotary_factor')
    rt=c2.rope_parameters['rope_type']
    if rt=='default':
        base=c2.rope_parameters['rope_theta']; dim=int(cfg['head_dim']*c2.rope_parameters.get('partial_rotary_factor',1.0))
        inv=1.0/(base**(torch.arange(0,dim,2,dtype=torch.int64).float()/dim)); sc=1.0
    else:
        inv,sc=ROPE_INIT_FUNCTIONS[rt](c2,None)
    res[name]=(inv.double(),float(sc))
    print(f"{name:6s} n={inv.numel():3d} [0]={inv[0].item():.9g} [1]={inv[1].item():.9g} [last]={inv[-1].item():.9g} attn_scale={sc!r}")
    # our python formula
    rp=cfg['rope_parameters'][key]
    if rt=='yarn':
        mine,_=yarn_inv_freq(cfg['head_dim'],rp.get('partial_rotary_factor',1.0),rp['rope_theta'],rp['factor'],
                             rp['original_max_position_embeddings'],rp['beta_fast'],rp['beta_slow'])
        print(f"       python-formula maxdiff = {(inv.double()-mine.double()).abs().max().item():.3e}")
json.dump({k:{'inv':v[0].tolist(),'scale':v[1]} for k,v in res.items()},open('docs/golden/rope_ref.json','w'))
print("wrote docs/golden/rope_ref.json")
