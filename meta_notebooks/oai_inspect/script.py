import numpy as np, os, glob
for root, dirs, files in os.walk("/kaggle/input"):
    print("DIR", root, "->", files[:5], flush=True)
files = glob.glob("/kaggle/input/**/*.npy", recursive=True)
print("npy files:", files, flush=True)
for f in files:
    z = np.load(f, mmap_mode="r")
    print(os.path.basename(f), "shape", z.shape, "dtype", z.dtype, flush=True)
    slab = z[0]
    print("  study0 min/max/mean:", float(slab.min()), float(slab.max()), float(slab.mean()), flush=True)
print("done")
