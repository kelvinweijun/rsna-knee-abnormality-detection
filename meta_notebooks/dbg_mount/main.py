import os, glob
print('L1', os.listdir('/kaggle/input'), flush=True)
for r, ds, fs in os.walk('/kaggle/input'):
    print('W', r, len(ds), fs[:3], flush=True)
    if r.count(os.sep) - 2 > 3: ds[:] = []
print('G', glob.glob('/kaggle/input/**/*.tar', recursive=True), flush=True)
