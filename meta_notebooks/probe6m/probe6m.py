import os
import sys

print('py', sys.version.split()[0])
import torch
import timm
print('torch', torch.__version__, '| timm', timm.__version__)
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
try:
    with open('/proc/meminfo') as f:
        for ln in f.read().splitlines()[:3]:
            print('MEM', ln)
except Exception as e:
    print('meminfo err', e)

cands = [
    'maxvit_base_tf_384', 'maxvit_large_tf_384',
    'convnext_v2_b.fcmae_in22k_in1k',
    'convnextv2_base.fcmae_ft_in22k_in1k',
    'convnext_base.fb_in22k_ft_in1k_384',
    'convnext_large_384_in22ft1k',
    'convnext_large.fb_in22k_ft_in1k_384',
    'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',
    'coatnet_2_rw_384.sw_in12k_ft_in1k',
    'tf_efficientnetv2_l.in21k_in1k',
    'tf_efficientnetv2_l.in21k_ft_in1k',
]
for c in cands:
    try:
        m = timm.create_model(c, pretrained=False)
        print('OK  ', c, '%.1fM' % (sum(p.numel() for p in m.parameters()) / 1e6))
        del m
    except Exception as e:
        print('FAIL', c, str(e)[:70])

print('convnextv2:', [n for n in timm.list_models('*convnextv2*') if 'base' in n][:10])
print('coatnet384:', [n for n in timm.list_models('*coatnet*384*')][:10])
print('effv2l:', list(timm.list_models('tf_efficientnetv2_l*'))[:8])
print('input files:')
for d, _, fs in os.walk('/kaggle/input'):
    for n in fs:
        if n in ('all_vols.npy', 'extra_vols.npy', 'fused_labels.csv', 'dataset-metadata.json'):
            print('  ', os.path.join(d, n))
