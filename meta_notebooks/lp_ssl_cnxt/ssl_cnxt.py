import glob, math, os, random, time
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

IN = "/kaggle/input"
OUT = "/kaggle/working"
FILES = sorted(glob.glob(IN + "/**/*.npz", recursive=True))
FILES = [f for f in FILES if "rsna" in f.lower() or True]
print("npz files", len(FILES), flush=True)
if not FILES:
    raise SystemExit("no npz found")
z0 = np.load(FILES[0]); k0 = list(z0.keys()); print("keys", k0, "shape", z0[k0[0]].shape, flush=True)

STEPS = 6000
BS = 64
LR = 2e-4
WD = 0.05
TEMP = 0.1
RES = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def load_two(path, rng):
    z = np.load(path)
    v = z["data"] if "data" in z else z[list(z.keys())[0]]
    if v.ndim == 4:
        v = v[0]
    D = v.shape[0]
    if D < 9:
        return None
    c1 = int(rng.integers(2, D - 2))
    c2 = int(rng.integers(2, D - 2))
    for _ in range(8):
        if abs(c1 - c2) >= 3:
            break
        c2 = int(rng.integers(2, D - 2))
    if abs(c1 - c2) < 3:
        return None
    a = torch.from_numpy(np.stack([v[c1 - 1], v[c1], v[c1 + 1]], 0).astype(np.float32) / 255.0)
    b = torch.from_numpy(np.stack([v[c2 - 1], v[c2], v[c2 + 1]], 0).astype(np.float32) / 255.0)
    return a, b

def aug(x, g):
    # x: (3,H,W) [0,1]
    _, H, W = x.shape
    s = float(torch.empty(1).uniform_(0.2, 0.5))
    ar = math.exp(float(torch.empty(1).uniform_(math.log(0.8), math.log(1.25))))
    ch = int(round(math.sqrt(s * H * W / ar))); cw = int(round(ch * ar))
    ch = min(ch, H); cw = min(cw, W)
    y0 = int(torch.randint(0, H - ch + 1, (1,))); x0 = int(torch.randint(0, W - cw + 1, (1,)))
    x = x[:, y0:y0+ch, x0:x0+cw]
    x = F.interpolate(x[None], size=(RES, RES), mode="bilinear", align_corners=False)[0]
    if g.random() < 0.5: x = torch.flip(x, dims=[2])
    if g.random() < 0.2: x = torch.flip(x, dims=[1])
    if g.random() < 0.8:
        x = (x * float(torch.empty(1).uniform_(0.7, 1.3)) + float(torch.empty(1).uniform_(-0.1, 0.1))).clamp(0, 1)
    if g.random() < 0.3:
        k = 3 + 2 * int(g.integers(0, 2))
        x = F.avg_pool2d(F.pad(x[None], (k//2, k//2, k//2, k//2), mode="reflect"), k, 1)[0]
    if g.random() < 0.2: x = (x + torch.randn_like(x) * 0.03).clamp(0, 1)
    x = (x - MEAN[0]) / STD[0]
    if g.random() < 0.2:
        eh = int(RES * float(torch.empty(1).uniform_(0.1, 0.25))); ew = int(RES * float(torch.empty(1).uniform_(0.1, 0.25)))
        y0 = int(torch.randint(0, RES - eh, (1,))); x0 = int(torch.randint(0, RES - ew, (1,)))
        x = x.clone(); x[:, y0:y0+eh, x0:x0+ew] = 0
    return x

class DS(torch.utils.data.Dataset):
    def __init__(self, files): self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        g = np.random.default_rng(random.getrandbits(32))
        try:
            pair = load_two(self.files[i], g)
        except Exception:
            return self.__getitem__((i + 1) % len(self))
        if pair is None:
            return self.__getitem__((i + 1) % len(self))
        return aug(pair[0], g), aug(pair[1], g)

def nt_xent(a, b, t):
    z = torch.cat([a, b], 0)
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / t
    n = z.shape[0]
    sim.fill_diagonal_(-1e4)
    tgt = torch.cat([torch.arange(n // 2, n), torch.arange(0, n // 2)]).to(z.device)
    return F.cross_entropy(sim, tgt)

def main():
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    bb = timm.create_model("convnext_small.fb_in22k_ft_in1k", pretrained=True, num_classes=0, global_pool="avg")
    dim = bb.num_features
    proj = nn.Sequential(nn.Linear(dim, 1024), nn.GELU(), nn.Linear(1024, 128))
    bb, proj = bb.to(dev), proj.to(dev)
    opt = torch.optim.AdamW(list(bb.parameters()) + list(proj.parameters()), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300.0) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / STEPS))))
    scaler = torch.amp.GradScaler("cuda")
    ds = DS(FILES)
    dl = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=True, num_workers=4, drop_last=True,
                                     persistent_workers=True, prefetch_factor=4)
    it = iter(dl)
    t0 = time.time(); run = 0.0
    for step in range(1, STEPS + 1):
        try:
            v1, v2 = next(it)
        except StopIteration:
            it = iter(dl); v1, v2 = next(it)
        v1, v2 = v1.to(dev, non_blocking=True), v2.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            za = proj(bb(v1)); zb = proj(bb(v2))
            loss = nt_xent(za.float(), zb.float(), TEMP)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        run += float(loss.detach())
        if step % 50 == 0:
            print(f"step {step} loss {run/50:.4f} lr {sch.get_last_lr()[0]:.2e} elapsed {time.time()-t0:.0f}s", flush=True)
            run = 0.0
        if step % 1000 == 0:
            sd = {"student": {"backbone." + k: v.detach().cpu() for k, v in bb.named_parameters()}, "epoch": step}
            torch.save(sd, os.path.join(OUT, f"ssl_cnxt_s224_step{step}.pt"))
            torch.save({"bb": bb.state_dict(), "proj": proj.state_dict(), "step": step}, os.path.join(OUT, f"ssl_full_step{step}.pt"))
            print("saved", step, flush=True)
    print("SSL_DONE", flush=True)

if __name__ == "__main__":
    main()
