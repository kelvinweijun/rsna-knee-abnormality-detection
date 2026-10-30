# %% [code]
# %% [code]
#!/usr/bin/env python3
"""Knee MRI: training the twelve-finding model

This is the training half of the model behind the public 0.924 inference notebook. It reads the
precomputed slice stacks, trains a CoAtNet backbone with a per-finding attention pooling head, and
writes a checkpoint you can drop straight into that notebook.

WHAT YOU NEED ATTACHED

  1. The preprocessed corpus, both parts:
       kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus          (3,200 studies)
       kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus-ext      (1,207 studies)
     Every study is already reduced to a fixed 44 x 336 x 336 uint8 stack, so no DICOM reading
     happens here. The two parts concatenate in order.

  2. The competition data, for train.csv.

  3. Training labels, as a parquet with a StudyInstanceUID column and the twelve finding columns.
     THIS IS NOT PROVIDED, and it is the one thing you have to bring yourself. See below.

THE LABEL PROBLEM, WHICH IS THE REAL PROBLEM

The competition gives you 4,407 studies and structured labels for only 58 of them. Every other
study carries a free-text radiology report and nothing else. So before any of this trains, you need
to turn 4,349 reports into twelve numbers each.

The approach behind the published weights was to read each report with a language model and emit
twelve probabilities rather than twelve yes or no answers: a report that hedges, saying a tear is
suspected, becomes something near 0.8 rather than a 1. Soft targets are far more forgiving than
forcing every hedged sentence into a hard label, and the loss here expects them. The 58 studies
that come with real labels are held out and used only for validation, never trained on.

Point --labels at your own parquet built that way. The format is one row per study: a
StudyInstanceUID column plus the twelve finding columns, values between 0 and 1.

WHAT THE MODEL DOES

Three neighbouring slices are stacked into the three channels of one image, so the network sees a
little of what lies above and below the middle slice: most of the benefit of a 3D model at the cost
of a 2D one. Each of these three-slice windows goes through the backbone, and the windows are then
pooled by an attention layer that has separate weights for each of the twelve findings. That last
part matters more than anything else here. A cruciate tear may be visible on two slices while
osteoarthritis spreads across many, and one shared pooling weight forces those to compete; giving
each finding its own attention lets each draw on the slices that actually show it.

Training samples k windows per study at random and evaluates on k_eval windows spread evenly, so
each epoch sees a different view of the same study. Nothing else is augmented.

At the end it keeps the best epoch by validation macro-AUC, and also writes a checkpoint that
averages the weights of the best three epochs. Weight averaging costs nothing at inference, unlike
averaging predictions from three models, and it usually gives a small gain.

TYPICAL RUN

  python train_knee.py --arch convnext_small.fb_in22k_ft_in1k --res 384 --epochs 16
      --bs 8 --k 12 --k_eval 24 --grad_ckpt --tag mymodel --labels /kaggle/input/YOURS/labels.parquet

About three hours on one 4090 for 16 epochs at 384. --grad_ckpt trades a little speed for a lot of
memory and is what makes bs 8 fit on a 24 GB card. Use --smoke for a fast wiring check.

MEMORY, MEASURED

The CoAtNet at 384 with --bs 8 and --grad_ckpt sits at about 23.7 GB, so it needs a 24 GB card.
On 16 GB use --bs 2 or 4. Separately from the GPU, this script keeps the best checkpoint plus the
top --topk epochs' weights in system RAM so it can average them at the end; that is about 87 MB a
copy for the small ViT and 293 MB for the CoAtNet, and the list fills over the first few epochs.
If you run out of host memory a few epochs in, set --topk 1 and lower --workers.

The checkpoint it writes is a dict with keys model, arch, res and lab, which is exactly what the
inference notebook expects.
"""
import os, sys, time, json, math, random, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import timm

# --- ROI localizer (anatomical joint crop). Optional so the no-ROI path is untouched. ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from roi_localize import square_box as _roi_square_box, compartments as _roi_compartments
except Exception:
    _roi_square_box = _roi_compartments = None

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
HERE = os.path.dirname(os.path.abspath(__file__))
if not os.access(HERE, os.W_OK):
    HERE = "/kaggle/working"   # read-only /kaggle/src; checkpoints must land in working
RSNA = os.path.dirname(HERE)

# ---------------------------------------------------------------------------
# Input discovery. On Kaggle the corpus arrives as two read-only datasets and the
# competition data as a third, so nothing lives beside this script. Everything in
# this block is discovery only - the training code further down is unchanged.
# ---------------------------------------------------------------------------
def _find(*names, root=os.environ.get("RAPTOR_INPUT", "/kaggle/input")):
    """First path under root whose basename matches one of names."""
    for d, _, fs in os.walk(root):
        for n in names:
            if n in fs:
                return os.path.join(d, n)
    return None


class _TwoPartVols:
    """Presents the two published corpus parts as one array of shape (4407, 44, 336, 336).

    Both parts stay memory-mapped and are never concatenated on disk: copying 22 GB would be
    pointless when every read is a single study. Row order is part 1 then part 2, matching the
    order the id files concatenate in. That ordering is the contract between volumes, masks and
    ids, so do not sort any of them independently.
    """
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = a.shape[0]
        self.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]


def _open_corpus():
    """Return (vols, masks), from a local single-file corpus or the two public parts."""
    local_v = os.path.join(HERE, "all_vols.npy")
    if os.path.exists(local_v):
        return (np.load(local_v, mmap_mode="r"),
                np.load(os.path.join(HERE, "all_masks.npy")))
    av, bv = _find("all_vols.npy"), _find("extra_vols.npy")
    am, bm = _find("all_masks.npy"), _find("extra_masks.npy")
    if not all((av, bv, am, bm)):
        raise SystemExit("Could not find the corpus. Attach both parts: "
                         "dreaddevelopment/knee-raptor-corpus and "
                         "dreaddevelopment/knee-raptor-corpus-ext")
    vols = _TwoPartVols(np.load(av, mmap_mode="r"), np.load(bv, mmap_mode="r"))
    masks = np.concatenate([np.load(am), np.load(bm)], axis=0)
    return vols, masks


def _open_ids():
    local = os.path.join(HERE, "all_ids.npy")
    if os.path.exists(local):
        return np.load(local, allow_pickle=True).astype(str)
    a, b = _find("all_ids.npy"), _find("extra_ids.npy")
    if not (a and b):
        raise SystemExit("Could not find all_ids.npy / extra_ids.npy - attach both corpus parts.")
    return np.concatenate([np.load(a, allow_pickle=True).astype(str),
                           np.load(b, allow_pickle=True).astype(str)])
LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA",
       "Effusion","Synovitis","Baker's","Contusion","Fracture"]

_GOLD_CSV = r"""StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
1.2.826.0.1.3680043.8.498.10095687747295410396510538520594649149,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.10170898615867673028696505248839028269,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.10306159113324811538703788080836752052,0.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.11287937729196958426538087439102017580,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.11382021393803389951964005983002209238,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.11548045715264151632153040089882701935,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0
1.2.826.0.1.3680043.8.498.11557620559191469069130827959098335840,1.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.11771393824519892797114773408583976756,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0
1.2.826.0.1.3680043.8.498.11851412923016044948101698015974810604,0.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0
1.2.826.0.1.3680043.8.498.11915937982684988073644209606907169581,0.0,0.0,0.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0
1.2.826.0.1.3680043.8.498.12448079646359892252441208258836556945,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.12505035424093604269515328931488770819,1.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0
1.2.826.0.1.3680043.8.498.12606657226568558340797193167488111973,1.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.12801308844398614687904447633432197492,0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0
1.2.826.0.1.3680043.8.498.12978510157202852202776899910529174803,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0
1.2.826.0.1.3680043.8.498.13267780356245120052411517053322874891,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0
1.2.826.0.1.3680043.8.498.13335129881737731410081002627729200903,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0
1.2.826.0.1.3680043.8.498.15593638897292057356864060466120253309,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.16060119389060497136231217718921482192,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,1.0
1.2.826.0.1.3680043.8.498.17844546765907321649997094867791102711,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0
1.2.826.0.1.3680043.8.498.18392509497170616983977319528036573378,1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0
1.2.826.0.1.3680043.8.498.22109739962224309418874538994436903404,0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0
1.2.826.0.1.3680043.8.498.25695966186129395049687747156570466645,0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.26790702379506190936834447203448882465,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0
1.2.826.0.1.3680043.8.498.27437263843446879932281897446100134924,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.28925345859498351203477642741908452608,0.0,0.0,1.0,1.0,1.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0
1.2.826.0.1.3680043.8.498.29764868091238287072166823522853419550,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0
1.2.826.0.1.3680043.8.498.30246079718471552972130572444383079911,1.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0
1.2.826.0.1.3680043.8.498.32321830776739689645700555055955725945,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0,1.0
1.2.826.0.1.3680043.8.498.37040910196459054543310335064167212742,1.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0
1.2.826.0.1.3680043.8.498.37392127307867238524251864751713143683,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0
1.2.826.0.1.3680043.8.498.39500553372517290823815606829518305500,0.0,0.0,0.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.41600498783384864111179175518614837125,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0
1.2.826.0.1.3680043.8.498.47921753480592595198052407850568677187,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0
1.2.826.0.1.3680043.8.498.48348615349418615469875801439348424274,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.48946580946665031852355005294734101132,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0
1.2.826.0.1.3680043.8.498.54977581323931277817074741182670080450,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.56512564464301544960446419143135447363,0.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.59483999067816153759785789654766826710,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0
1.2.826.0.1.3680043.8.498.62465595376489211274225312453216559395,1.0,1.0,1.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,0.0,1.0
1.2.826.0.1.3680043.8.498.62549354677638403845904556149868367236,1.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0
1.2.826.0.1.3680043.8.498.64408609256163127278435484217683272910,0.0,0.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0
1.2.826.0.1.3680043.8.498.64703506772167798469048460791472465039,0.0,1.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0
1.2.826.0.1.3680043.8.498.65708905118633339771181857781063784327,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0
1.2.826.0.1.3680043.8.498.67188121544063723837669983597453941774,1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.69392348385274125385290015404144639002,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.72853333220043794904856138561095171921,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.73527530686853911124431549317032662220,1.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0
1.2.826.0.1.3680043.8.498.73926443729786165628848843707532839995,0.0,0.0,1.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.75187434248356774277526985329346125190,1.0,0.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0,0.0
1.2.826.0.1.3680043.8.498.77362718298550276679350855963451855003,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0
1.2.826.0.1.3680043.8.498.78512215519177850923279235346968676828,0.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.82166943552764439138333504456139890254,1.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.86968600239724310678905311244945464037,1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.88077418639301174409926781329613570435,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0
1.2.826.0.1.3680043.8.498.90283565381042081768587894596970552767,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0
1.2.826.0.1.3680043.8.498.94433753471890306108089557761231739312,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0
1.2.826.0.1.3680043.8.498.97274720257634584071500649275217521662,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0"""


_NO_LABELS = """
No training labels found, so there is nothing to train against.

The competition labels only 58 of the 4,407 studies. The other 4,349 carry a free-text
radiology report instead, so before this can train you have to turn those reports into
twelve probabilities per study and pass the result with --labels.

Expected format: a parquet with a StudyInstanceUID column plus the columns
  {cols}
with values between 0 and 1. Soft values work better than hard 0/1 here: the loss is built
for them, and hedged reports are common.

Everything else in this notebook is ready to run once that file exists.
"""


# ------------------------------- data ----------------------------------------
class StudyWindows(Dataset):
    """Per-study bag of 2.5D windows sampled from all_vols.npy (memmap).
    Each window = 3 physically-consecutive slices -> RGB, resized to `res`, in [0,1]
    (matches the SSL input pipeline: ToTensor, no ImageNet norm)."""
    def __init__(self, root, ids, id2row, labels, res, k, train, aug=True, norm="none",
                 roi=False, roi_mode="tight", roi_pad=0.06, roi_overlap=0.12,
                 roi_sbox=None, roi_cen=None):
        self.root = root
        self.ids = ids
        self.id2row = id2row
        self.labels = labels              # dict uid -> np.float32[12]
        self.res, self.k, self.train, self.aug = res, k, train, aug
        self.norm = norm                  # "none"=[0,1] (Raptor SSL); "imagenet"=DINOv2 stats
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.vols = None; self.masks = None
        # --- ROI anatomical joint-crop config ---
        self.roi = bool(roi)
        self.roi_mode, self.roi_pad, self.roi_overlap = roi_mode, roi_pad, roi_overlap
        self.roi_sbox = roi_sbox          # (N,D,4) int16 per-slice tissue bbox, or None
        self.roi_cen = roi_cen            # (N,D,2) int16 per-slice joint centroid, or None
        if self.roi and (_roi_square_box is None or roi_sbox is None):
            raise RuntimeError("roi=True but roi_localize or roi_boxes not available")

    def __len__(self): return len(self.ids)

    def _ensure(self):
        if self.vols is None:
            self.vols, self.masks = _open_corpus()   # (N,D,H,W) view, (N,D) u8

    def _centers(self, valid, count):
        # valid slice indices; window centers must have both neighbors valid & in-range
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
        if not cs: cs = [max(1, min((lo + hi) // 2, self._D - 2))]
        if self.train:
            reps = count // len(cs) + 1
            pool = (cs * reps)
            random.shuffle(pool)
            return pool[:count]
        # eval: evenly spaced deterministic
        idx = np.linspace(0, len(cs) - 1, count).round().astype(int)
        return [cs[i] for i in idx]

    def _resize(self, tri):
        """tri: (3,h,w) float32 [0,1] -> (3,res,res) float32."""
        t = torch.from_numpy(np.ascontiguousarray(tri))
        if t.shape[-1] != self.res or t.shape[-2] != self.res:
            t = F.interpolate(t[None], size=(self.res, self.res), mode="bilinear",
                              align_corners=False)[0]
        return t.numpy()

    def __getitem__(self, i):
        self._ensure()
        uid = self.ids[i]; row = self.id2row[uid]
        self._D = self.vols.shape[1]
        m = self.masks[row]
        valid = np.where(m > 0)[0]
        if len(valid) < 3: valid = np.arange(min(3, self._D))
        # compartment mode emits 2 crops/center -> sample ceil(k/2) centers to keep #windows==k
        compart = self.roi and self.roi_mode == "compartment"
        n_centers = (self.k + 1) // 2 if compart else self.k
        cs = self._centers(valid, n_centers)
        vol = self.vols[row]  # (D,H,W) u8  (single study read)
        tiles = []            # list of (3,res,res) float32
        for c in cs:
            c = max(1, min(c, self._D - 2))
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0  # (3,H,W)
            H, W = tri.shape[-2], tri.shape[-1]
            if not self.roi:
                tiles.append(self._resize(tri))
                continue
            # one joint box from the CENTER slice, applied to all 3 slices (keeps RGB registered)
            sq_mode = "tight" if compart else self.roi_mode
            sq = _roi_square_box(tuple(int(v) for v in self.roi_sbox[row, c]), W=W, H=H,
                                 pad=self.roi_pad, mode=sq_mode,
                                 centroid=tuple(float(v) for v in self.roi_cen[row, c]))
            if compart:
                left, right = _roi_compartments(sq, overlap=self.roi_overlap)
                for box in (left, right):
                    x0, y0, x1, y1 = box
                    tiles.append(self._resize(tri[:, y0:y1, x0:x1]))
            else:
                x0, y0, x1, y1 = sq
                tiles.append(self._resize(tri[:, y0:y1, x0:x1]))
        if len(tiles) > self.k:
            tiles = tiles[:self.k]
        wins = np.stack(tiles, 0)          # (K,3,res,res)
        x = torch.from_numpy(wins)
        if self.train and self.aug:
            # light medical-safe aug: NO flips (laterality is signal); mild intensity jitter
            g = 1.0 + (random.random() - 0.5) * 0.20
            x = (x * g).clamp(0, 1)
        if self.norm == "imagenet":        # each backbone at its correct input distribution
            x = (x - self._mean) / self._std
        y = torch.from_numpy(self.labels[uid])
        return x, y


def collate(batch):
    xs = torch.stack([b[0] for b in batch])   # (B,K,3,res,res)
    ys = torch.stack([b[1] for b in batch])   # (B,12)
    return xs, ys


class OAIDataset(Dataset):
    """OAI 3D-DESS knees: (N,128,128,64) float32 per file (axis0=slices, axis2=W thin -> pad).
    File name gives a binary abnormal/normal label -> weak OA supervision, everything else
    masked out via per-element BCE weight."""
    def __init__(self, res, k, train=True, norm="imagenet", n_max=0):
        self.res, self.k, self.train, self.norm = res, k, train, norm
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.vols = []; self.targets = []
        for fn, lab in (("normal-3DESS-128-64.npy", 0.0), ("abnormal-3DESS-128-64.npy", 1.0)):
            p = _find(fn)
            if not p:
                raise SystemExit(f"OAI file not mounted: {fn}")
            z = np.load(p, mmap_mode="r")
            for r in range(z.shape[0]):
                y = np.zeros(12, dtype=np.float32)
                y[4] = lab; y[5] = lab          # Medial OA, Lateral OA
                m = np.zeros(12, dtype=np.float32)
                m[4] = 1.0; m[5] = 0.5
                self.vols.append((z, r)); self.targets.append((y, m))
        if n_max:
            self.vols = self.vols[:n_max]; self.targets = self.targets[:n_max]
        random.shuffle(self.targets) if False else None

    def __len__(self): return len(self.vols)

    def _resize(self, tri):
        t = torch.from_numpy(np.ascontiguousarray(tri))
        if t.shape[-1] != self.res or t.shape[-2] != self.res:
            t = F.interpolate(t[None], size=(self.res, self.res), mode="bilinear",
                              align_corners=False)[0]
        return t.numpy()

    def __getitem__(self, i):
        z, r = self.vols[i]
        vol = np.asarray(z[r])                       # (128,128,64) float32
        lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
        vol = np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
        D = vol.shape[0]
        if self.train:
            cs = random.sample(range(1, D - 1), min(self.k, D - 2))
        else:
            cs = np.linspace(1, D - 2, self.k).round().astype(int).tolist()
        tiles = []
        for c in cs:
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0)
            tri = np.pad(tri, ((0, 0), (0, 0), (32, 32)), mode="constant")
            tiles.append(self._resize(tri))
        x = torch.from_numpy(np.stack(tiles, 0))
        if self.train:
            g = 1.0 + (random.random() - 0.5) * 0.20
            x = (x * g).clamp(0, 1)
        if self.norm == "imagenet":
            x = (x - self._mean) / self._std
        y, m = self.targets[i]
        return x, torch.from_numpy(y), torch.from_numpy(m)


class MoaksDataset(Dataset):
    """MOAKS-labelled OAI knees written by moaks_convert.py.

    One npz per knee: vol uint8 (128,128,64), y (12,), m (12,). Same window
    sampling and geometry as OAIDataset, but labels come from MOAKS grades."""
    def __init__(self, root, res, k, train=True, norm="none", n_max=0, soft=None):
        self.res, self.k, self.train, self.norm = res, k, train, norm
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.files = sorted(os.path.join(root, f) for f in os.listdir(root)
                            if f.endswith(".npz"))
        self.soft = soft or {}
        if n_max:
            self.files = self.files[:n_max]
        if not self.files:
            raise SystemExit(f"no MOAKS npz shards in {root}")

    def __len__(self): return len(self.files)

    def _resize(self, tri):
        t = torch.from_numpy(np.ascontiguousarray(tri))
        if t.shape[-1] != self.res or t.shape[-2] != self.res:
            t = F.interpolate(t[None], size=(self.res, self.res), mode="bilinear",
                              align_corners=False)[0]
        return t.numpy()

    def __getitem__(self, i):
        z = np.load(self.files[i])
        vol = z["vol"].astype(np.float32) / 255.0
        lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
        vol = np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
        D = vol.shape[0]
        if self.train:
            cs = random.sample(range(1, D - 1), min(self.k, D - 2))
        else:
            cs = np.linspace(1, D - 2, self.k).round().astype(int).tolist()
        tiles = []
        for c in cs:
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0)
            tri = np.pad(tri, ((0, 0), (0, 0), (32, 32)), mode="constant")
            tiles.append(self._resize(tri))
        x = torch.from_numpy(np.stack(tiles, 0))
        if self.train:
            g = 1.0 + (random.random() - 0.5) * 0.20
            x = (x * g).clamp(0, 1)
        if self.norm == "imagenet":
            x = (x - self._mean) / self._std
        y = z["y"].astype(np.float32).copy()
        m = z["m"].astype(np.float32).copy()
        if self.soft:
            _s = self.soft.get(os.path.basename(self.files[i]))
            if _s:
                for _j, _v in _s.items():
                    if m[_j] > 0:
                        y[_j] = _v
        keep = np.zeros(12, dtype=np.float32); keep[[5, 6, 8]] = 1.0
        m = m * keep
        return x, torch.from_numpy(y), torch.from_numpy(m)


def _load_soft(path):
    """file -> {label_index: soft target} for MOAKS-grade supervision."""
    if not path or not os.path.exists(path):
        return {}
    import csv as _csv
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in _csv.DictReader(f):
            d = {}
            for col, idx in (("soft_LatOA", 5), ("soft_PFOA", 6), ("soft_MedOA", 4)):
                v = (r.get(col) or "").strip()
                if v not in ("", "."):
                    d[idx] = float(v)
            if d:
                out.setdefault(r["file"], {}).update(d)
    return out


def collate_oai(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]))


# ------------------------------- model ---------------------------------------
def build_backbone(arch="vit_small_patch16_224", pretrained=False, img_size=None):
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if img_size is not None:
        kw.update(img_size=img_size)
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    try:
        return timm.create_model(arch, **kw)
    except TypeError:
        kw.pop("img_size", None)
        return timm.create_model(arch, **kw)


def load_raptor(bb, ckpt_path):
    if ckpt_path in ("timm", "pretrained"):
        return "timm-pretrained"
    if ckpt_path in ("", "none", "None"):
        print("[raptor] RANDOM-INIT control (no SSL weights)", flush=True)
        return "random-init"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st = ck["student"] if "student" in ck else ck
    bbst = {k[len("backbone."):]: v for k, v in st.items() if k.startswith("backbone.")}
    missing, unexpected = bb.load_state_dict(bbst, strict=False)
    ep = ck.get("epoch", "?")
    print(f"[raptor] loaded backbone from {os.path.basename(ckpt_path)} (ssl epoch {ep}) | "
          f"loaded {len(bbst)} tensors, missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    return f"{os.path.basename(ckpt_path)}@ep{ep}"


class RaptorClassifier(nn.Module):
    """Raptor encoder + per-diagnosis attention-MIL head (12 findings)."""
    def __init__(self, backbone, F_dim=384, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))


# ------------------------------- train ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # Defaults to ImageNet-pretrained weights, which is what every released arm was trained from.
    # The previous default pointed at a self-supervised checkpoint that is not distributed here,
    # so following the example run above failed on a missing file.
    ap.add_argument("--ckpt", default="timm",
                    help="'timm' for ImageNet-pretrained weights, or a path to your own "
                         "self-supervised checkpoint")
    ap.add_argument("--init", default="",
                    help="full model checkpoint (dict with a 'model' state dict) to initialize from")
    ap.add_argument("--arch", default="vit_small_patch16_224")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--k_eval", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--bb_lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--freeze_blocks", type=int, default=0)
    # Each kept epoch is a full copy of the weights in system RAM (~87 MB small ViT, ~293 MB
    # CoAtNet). Lower to 1 on a memory-tight machine; at 1 no averaged checkpoint is written.
    ap.add_argument("--topk", type=int, default=3,
                    help="top epochs kept for weight averaging (1 disables averaging)")
    ap.add_argument("--norm", default="none", choices=["none", "imagenet"])
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--labels", default=None,
                    help="parquet of training labels: StudyInstanceUID + the twelve finding "
                         "columns, values 0..1. Not provided with this notebook - see the header.")
    ap.add_argument("--oai_soft", default=None,
                    help="CSV of graded MOAKS soft targets (patient_id, knee_side, file, soft_LatOA, soft_PFOA, ...)")
    ap.add_argument("--oai_dir", default=None,
                    help="directory of MOAKS-converted per-knee npz (moaks_convert.py); "
                         "replaces the two binary OAI npy files with 9-label supervision")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    # ---- CV fold hook ----
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--fold_file", default=None)
    # ---- anatomical ROI joint-crop (A/B lever). Default OFF -> identical to baseline. ----
    ap.add_argument("--roi", action="store_true")
    ap.add_argument("--roi_mode", default="compartment", choices=["tight", "safe", "compartment"])
    ap.add_argument("--roi_pad", type=float, default=0.06)
    ap.add_argument("--roi_overlap", type=float, default=0.12)
    ap.add_argument("--roi_boxes", default=os.path.join(HERE, "roi_boxes.npz"))
    # isotriage 2xT4 run: bs4+grad_ckpt fits 16GB, coatnet384 recipe, JEV-hybrid labels.
    ap.set_defaults(arch="convnext_small.fb_in22k_ft_in1k", res=384, k=12, k_eval=24,
                    epochs=7, bs=4, topk=3, workers=4, tag="jev_oai384", grad_ckpt=True,
                    norm="imagenet", smoke=False)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        a.epochs, a.bs, a.k, a.k_eval, a.limit, a.workers = 2, 4, 4, 6, 40, 0
    print(f"device {dev} | res {a.res} | k {a.k}/{a.k_eval} | bs {a.bs} | tag {a.tag} | "
          f"roi={a.roi}({a.roi_mode})", flush=True)

    # ---- ids / labels ----
    ids = _open_ids()
    id2row = {u: i for i, u in enumerate(ids)}
    idset = set(ids)
    _tcsv = os.path.join(RSNA, "train.csv")
    if not os.path.exists(_tcsv):
        _tcsv = _find("train.csv")
    if _tcsv:
        tr = pd.read_csv(_tcsv)
    else:
        import io as _io
        print("[gold] train.csv not mounted - using embedded gold58", flush=True)
        tr = pd.read_csv(_io.StringIO(_GOLD_CSV))
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
    gold_ids = [u for u in gold_df.index if u in idset]
    _lab = (a.labels or _find("fused_labels.csv") or _find("HYBRID_labels_scores.csv")
            or _find("labels_llm_soft.parquet"))
    if not _lab or not os.path.exists(_lab):
        # Exit cleanly rather than as a failure: running this notebook as published, with no
        # labels attached, is the expected path and should read as an explanation, not a crash.
        print(_NO_LABELS.format(cols=", ".join(LAB)), flush=True)
        raise SystemExit(0)
    soft = pd.read_csv(_lab) if _lab.endswith(".csv") else pd.read_parquet(_lab)
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str); soft = soft.set_index("StudyInstanceUID")
    goldset = set(gold_ids)
    train_ids = [u for u in ids if u in soft.index and u not in goldset]
    # ---- CV fold hook: hold out fold `a.fold`, train on the rest ----
    oof_ids = []
    if a.folds > 0:
        assert 0 <= a.fold < a.folds, f"--fold must be in [0,{a.folds}) when --folds>0"
        if a.fold_file and os.path.exists(a.fold_file):
            fmap = json.load(open(a.fold_file))["folds"]
        else:
            pool = sorted(train_ids)
            rng = np.random.RandomState(1234)
            perm = rng.permutation(len(pool))
            fmap = {pool[idx]: int(rank % a.folds) for rank, idx in enumerate(perm)}
            print("[cv] fold_file absent - generated folds deterministically (seed 1234)", flush=True)
        held = set(u for u in train_ids if fmap.get(u, -1) == a.fold)
        oof_ids = [u for u in train_ids if u in held]
        train_ids = [u for u in train_ids if u not in held]
        print(f"[cv] fold {a.fold}/{a.folds}: train {len(train_ids)} | OOF held-out {len(oof_ids)} "
              f"| fold_file {os.path.basename(a.fold_file) if a.fold_file else 'generated'}", flush=True)
    if a.limit: train_ids = train_ids[:a.limit]
    labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in train_ids}
    for u in oof_ids: labels[u] = soft.loc[u, LAB].values.astype(np.float32)
    for u in gold_ids: labels[u] = gold_df.loc[u, LAB].values.astype(np.float32)
    print(f"train {len(train_ids)} | gold-val {len(gold_ids)}"
          + (f" | oof {len(oof_ids)}" if oof_ids else ""), flush=True)

    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)

    # ---- ROI boxes (only when --roi) ----
    roi_sbox = roi_cen = None
    if a.roi:
        rb = np.load(a.roi_boxes)
        rb_ids = rb["ids"].astype(str)
        if not np.array_equal(rb_ids, ids):
            rmap = {u: i for i, u in enumerate(rb_ids)}
            missing = [u for u in ids if u not in rmap]
            if missing:
                raise RuntimeError(f"roi_boxes missing {len(missing)} corpus ids (e.g. {missing[:2]})")
            order = np.array([rmap[u] for u in ids])
            roi_sbox = rb["sbox"][order]; roi_cen = rb["cen"][order]
        else:
            roi_sbox = rb["sbox"]; roi_cen = rb["cen"]
        print(f"[roi] ENABLED mode={a.roi_mode} pad={a.roi_pad} overlap={a.roi_overlap} "
              f"boxes={os.path.basename(a.roi_boxes)} sbox={roi_sbox.shape}", flush=True)
    _roi_kw = dict(roi=a.roi, roi_mode=a.roi_mode, roi_pad=a.roi_pad, roi_overlap=a.roi_overlap,
                   roi_sbox=roi_sbox, roi_cen=roi_cen)

    tds = StudyWindows(HERE, train_ids, id2row, labels, a.res, a.k, train=True, norm=a.norm, **_roi_kw)
    vds = StudyWindows(HERE, gold_ids, id2row, labels, a.res, a.k_eval, train=False, norm=a.norm, **_roi_kw)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    collate_fn=collate, pin_memory=True, persistent_workers=a.workers > 0)
    vl = DataLoader(vds, batch_size=max(2, a.bs // 2), shuffle=False, num_workers=a.workers,
                    collate_fn=collate, persistent_workers=a.workers > 0)

    use_timm = a.ckpt in ("timm", "pretrained")
    bb = build_backbone(a.arch, pretrained=use_timm, img_size=a.res)
    src = load_raptor(bb, a.ckpt)
    if use_timm: print(f"[raptor] timm-pretrained backbone: {a.arch}", flush=True)
    F_dim = bb.num_features
    if a.grad_ckpt:
        try:
            bb.set_grad_checkpointing(True); print("[raptor] gradient checkpointing ON", flush=True)
        except Exception as e:
            print(f"[raptor] grad_ckpt unsupported for {a.arch}: {e}", flush=True)
    model = RaptorClassifier(bb, F_dim=F_dim).to(dev)
    _init = _find(os.environ.get("RAPTOR_INIT", "raptor_ft_jev384_swa.pt"))
    if _init:
        _ick = torch.load(_init, map_location="cpu", weights_only=False)
        model.load_state_dict(_ick["model"], strict=True)
        print(f"[init] resumed from {_init} (prev gold {_ick.get('gold_auc')})", flush=True)
    else:
        print("[init] no init checkpoint mounted - training from timm-pretrained weights", flush=True)
    if a.init:
        _ick = torch.load(a.init, map_location="cpu", weights_only=False)
        model.load_state_dict(_ick.get("model", _ick), strict=True)
        print(f"[init] loaded full model from {os.path.basename(a.init)} "
              f"(prev gold {_ick.get('gold_auc')})", flush=True)
    if a.freeze_blocks > 0:
        # Backbones name their layer groups differently: ViT-style models use blocks.N., while
        # CoAtNet, MaxViT and ConvNeXt use stages.N. Matching only one of those froze nothing at
        # all on the other family, and did it silently. Try both, and always say what happened.
        pref = tuple([f"blocks.{b}." for b in range(a.freeze_blocks)]
                     + [f"stages.{b}." for b in range(a.freeze_blocks)])
        frozen = 0
        for nm, p in model.backbone.named_parameters():
            if nm.startswith(pref):
                p.requires_grad = False
                frozen += p.numel()
        n_t = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        if frozen:
            print(f"[raptor] froze the first {a.freeze_blocks} layer group(s): "
                  f"{n_t} tensors, {frozen/1e6:.2f}M parameters", flush=True)
        else:
            print(f"[raptor] WARNING: --freeze_blocks {a.freeze_blocks} matched no parameters on "
                  f"{a.arch}; every parameter is still being trained", flush=True)

    head_params = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.") and p.requires_grad]
    bb_params = [p for n_, p in model.named_parameters() if n_.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": a.bb_lr},
                             {"params": head_params, "lr": a.head_lr}], weight_decay=a.wd)
    steps = max((len(tl) // max(1, a.accum)) * a.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr],
                                                total_steps=steps, pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    if a.oai_dir:
        ods = MoaksDataset(a.oai_dir, res=a.res, k=a.k, train=True, norm=a.norm,
                           n_max=8 if a.smoke else 0, soft=_load_soft(a.oai_soft))
        print(f"[oai] soft targets {len(ods.soft)} knees from {a.oai_soft}", flush=True)
        print(f"[oai] MOAKS multisupervision from {a.oai_dir}", flush=True)
    else:
        ods = OAIDataset(res=a.res, k=a.k, train=True, norm=a.norm, n_max=8 if a.smoke else 0)
    otl = DataLoader(ods, batch_size=1, shuffle=True, collate_fn=collate_oai,
                     num_workers=1, pin_memory=True, drop_last=True)
    oit = iter(otl)
    _oai_every = 2   # concat one OAI batch into every 2nd RSNA step (~+50% epoch time)
    print(f"[oai] {len(ods)} OAI studies | "
          + ("3 MOAKS labels: PF OA / Lateral OA / Synovitis (nartaa recipe)" if a.oai_dir
             else "mask -> MedialOA(w=1) LateralOA(w=0.5)"), flush=True)

    @torch.no_grad()
    def evaluate():
        model.eval(); P = []; Y = []
        for x, y in vl:
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                o = torch.sigmoid(model(x.to(dev)).float())
            P.append(o.cpu().numpy()); Y.append(y.numpy())
        P = np.concatenate(P); Y = np.concatenate(Y)
        aucs = {}
        for j, name in enumerate(LAB):
            if len(set(Y[:, j].astype(int))) > 1:
                aucs[name] = float(roc_auc_score(Y[:, j], P[:, j]))
        return float(np.mean(list(aucs.values()))), aucs, P, Y

    try:
        scaler = torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
    best = 0.0; best_state = None; best_P = None; t0 = time.time(); hist = []
    TOPK = max(1, a.topk); topk = []
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for _bi, (x, y) in enumerate(tl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            n_r = x.shape[0]
            if _bi % _oai_every == 1:
                try:
                    xo, yo, mo = next(oit)
                except StopIteration:
                    oit = iter(otl); xo, yo, mo = next(oit)
                x = torch.cat([x, xo.to(dev, non_blocking=True)], 0)
                y = torch.cat([y, yo.to(dev, non_blocking=True)], 0)
                msk = torch.cat([torch.ones(n_r, 12, device=dev), mo.to(dev, non_blocking=True)], 0)
            else:
                msk = torch.ones(n_r, 12, device=dev)
            if _bi % max(1, a.accum) == 0:
                opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(
                    logits.float(), y, weight=msk, pos_weight=pw, reduction="none").mean()
            _acc = max(1, a.accum)
            scaler.scale(loss / _acc).backward()
            if (_bi + 1) % _acc == 0 or (_bi + 1) == len(tl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        au, aucs, P, Y = evaluate()
        hist.append({"ep": ep, "loss": tot / len(tl), "gold_auc": au})
        if au > best:
            best = au; best_P = P
            best_state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "gold_auc": au, "aucs": aucs, "src": src, "res": a.res,
                          "arch": a.arch, "lab": LAB, "epoch": ep}
        if au >= best and best_state is not None:
            _tmp = os.path.join(HERE, f"raptor_ft_{a.tag}.pt.tmp")
            torch.save(best_state, _tmp)
            os.replace(_tmp, os.path.join(HERE, f"raptor_ft_{a.tag}.pt"))
            np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}.npz"),
                     pred=best_P, truth=Y, ids=np.array(gold_ids))
            json.dump({"tag": a.tag, "src": src, "best_gold_auc": best,
                       "aucs": best_state["aucs"], "hist": hist, "res": a.res,
                       "epochs": a.epochs, "bb_lr": a.bb_lr, "head_lr": a.head_lr,
                       "n_train": len(train_ids), "n_gold": len(gold_ids),
                       "roi": a.roi, "roi_mode": a.roi_mode,
                       "partial": True, "epochs_done": ep + 1},
                      open(os.path.join(HERE, f"raptor_ft_{a.tag}.json"), "w"), indent=1)
            print(f"  [ckpt] best-so-far saved at ep{ep} ({best:.4f})", flush=True)
        if len(topk) < TOPK or au > min(t["gold_auc"] for t in topk):
            topk.append({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                         "gold_auc": au, "aucs": aucs, "src": src, "res": a.res,
                         "arch": a.arch, "lab": LAB, "epoch": ep, "P": P})
            topk.sort(key=lambda t: -t["gold_auc"])
            del topk[TOPK:]
        print(f"ep{ep} loss {tot/len(tl):.3f} | GOLD macro-AUC {au:.4f} (best {best:.4f}) | {time.time()-t0:.0f}s",
              flush=True)
        if time.time() - t0 > 39600:
            print("[time] 10h soft deadline reached - stopping cleanly with current best", flush=True)
            break
    _, aucs, _, Y = evaluate()
    print(f"\nDONE {src} | BEST GOLD macro-AUC {best:.4f}", flush=True)
    for k, v in (best_state["aucs"] if best_state else aucs).items():
        print(f"   {k:18s} {v:.3f}", flush=True)

    if best_state is not None:
        torch.save(best_state, os.path.join(HERE, f"raptor_ft_{a.tag}.pt"))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}.npz"),
                 pred=best_P, truth=Y, ids=np.array(gold_ids))
        json.dump({"tag": a.tag, "src": src, "best_gold_auc": best, "aucs": best_state["aucs"],
                   "hist": hist, "res": a.res, "epochs": a.epochs, "bb_lr": a.bb_lr,
                   "head_lr": a.head_lr, "n_train": len(train_ids), "n_gold": len(gold_ids),
                   "roi": a.roi, "roi_mode": a.roi_mode, "partial": False, "epochs_done": a.epochs},
                  open(os.path.join(HERE, f"raptor_ft_{a.tag}.json"), "w"), indent=1)
        print(f"saved raptor_ft_{a.tag}.pt / raptor_gold_{a.tag}.npz / raptor_ft_{a.tag}.json", flush=True)

    # ---- CV OOF: predict the held-out fold at the best (gold-selected) weights ----
    if a.folds > 0 and oof_ids and best_state is not None:
        model.load_state_dict(best_state["model"]); model.eval()
        ods = StudyWindows(HERE, oof_ids, id2row, labels, a.res, a.k_eval, train=False, norm=a.norm, **_roi_kw)
        ol = DataLoader(ods, batch_size=max(2, a.bs // 2), shuffle=False, num_workers=a.workers,
                        collate_fn=collate, persistent_workers=False)
        Po = []
        with torch.no_grad():
            for x, y in ol:
                with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                    o = torch.sigmoid(model(x.to(dev)).float())
                Po.append(o.cpu().numpy())
        Po = np.concatenate(Po)
        Yo = np.stack([labels[u] for u in oof_ids]).astype(np.float32)
        np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_fold{a.fold}.npz"),
                 pred=Po, truth=Yo, ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
        print(f"[cv] wrote raptor_oof_{a.tag}_fold{a.fold}.npz ({len(oof_ids)} studies; "
              f"truth = soft labels)", flush=True)

        # --- top-K epoch OOF (new) ---
        # The epoch-ensemble used to be judged on gold only; that gate is too small to
        # resolve the move. Re-run the held-out fold at each retained epoch instead.
        oof_by_ep = {best_state["epoch"]: Po}
        for t in topk:
            e = int(t["epoch"])
            if e in oof_by_ep:
                continue
            model.load_state_dict(t["model"]); model.eval()
            Pe = []
            with torch.no_grad():
                for x, y in ol:
                    with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                        Pe.append(torch.sigmoid(model(x.to(dev)).float()).cpu().numpy())
            Pe = np.concatenate(Pe); oof_by_ep[e] = Pe
            np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_ep{e}_fold{a.fold}.npz"),
                     pred=Pe, truth=Yo, ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
            print(f"[cv] wrote top-K epoch OOF ep{e}", flush=True)
        if len(oof_by_ep) > 1:
            Pens = np.mean(list(oof_by_ep.values()), axis=0)
            np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_epens_fold{a.fold}.npz"),
                     pred=Pens, truth=Yo, ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
            print(f"[cv] wrote epoch-ensemble OOF over epochs {sorted(oof_by_ep)}", flush=True)
        model.load_state_dict(best_state["model"]); model.eval()

    # --- weight-averaged checkpoint (new) ---
    if len(topk) > 1:
        import copy
        sds = [t["model"] for t in topk]
        avg = {}
        for k in sds[0]:
            v0 = sds[0][k]
            if v0.is_floating_point():
                avg[k] = sum(sd[k].double() for sd in sds).div(len(sds)).to(v0.dtype)
            else:
                avg[k] = v0.clone()          # e.g. num_batches_tracked
        swa_state = {"model": avg, "gold_auc": None, "aucs": {}, "src": src, "res": a.res,
                     "arch": a.arch, "lab": LAB, "epoch": [int(t["epoch"]) for t in topk],
                     "swa_over": [int(t["epoch"]) for t in topk]}
        model.load_state_dict(avg); model.eval()
        au_swa, aucs_swa, P_swa, Y_swa = evaluate()
        swa_state["gold_auc"] = au_swa; swa_state["aucs"] = aucs_swa
        torch.save(swa_state, os.path.join(HERE, f"raptor_ft_{a.tag}_swa.pt"))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}_swa.npz"),
                 pred=P_swa, truth=Y_swa, ids=np.array(gold_ids))
        print(f"SWA over epochs {[int(t['epoch']) for t in topk]} | gold {au_swa:.4f} "
              f"(best-epoch {best:.4f}) {'BETTER' if au_swa > best else 'no gain'}", flush=True)
        if a.folds > 0 and oof_ids:
            ods2 = StudyWindows(HERE, oof_ids, id2row, labels, a.res, a.k_eval, train=False,
                                norm=a.norm, **_roi_kw)
            ol2 = DataLoader(ods2, batch_size=max(2, a.bs // 2), shuffle=False,
                             num_workers=a.workers, collate_fn=collate, persistent_workers=False)
            Ps = []
            with torch.no_grad():
                for x, y in ol2:
                    with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                        Ps.append(torch.sigmoid(model(x.to(dev)).float()).cpu().numpy())
            Ps = np.concatenate(Ps)
            np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_swa_fold{a.fold}.npz"),
                     pred=Ps, truth=np.stack([labels[u] for u in oof_ids]).astype(np.float32),
                     ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
            print(f"[cv] wrote SWA OOF", flush=True)
        model.load_state_dict(best_state["model"]); model.eval()

    if len(topk) > 1:
        eps = [t["epoch"] for t in topk]
        for rank, t in enumerate(topk):
            if rank == 0:
                continue
            P_t = t.pop("P")
            np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}_ep{t['epoch']}.npz"),
                     pred=P_t, truth=Y, ids=np.array(gold_ids))
            t["P"] = P_t
        Pens = np.mean([t["P"] for t in topk], axis=0)
        ens_aucs = {}
        for j, name in enumerate(LAB):
            if len(set(Y[:, j].astype(int))) > 1:
                ens_aucs[name] = float(roc_auc_score(Y[:, j], Pens[:, j]))
        ens = float(np.mean(list(ens_aucs.values())))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}_epens.npz"),
                 pred=Pens, truth=Y, ids=np.array(gold_ids))
        print(f"TOPK epochs {eps} | best {best:.4f} | epoch-ensemble {ens:.4f} "
              f"({'BETTER' if ens > best else 'no gain'})", flush=True)


if __name__ == "__main__":
    import sys as _sys, glob as _glob, os as _os
    _hits = sorted(_glob.glob("/kaggle/input/**/*.npz", recursive=True))  # EB_UNTAR probe
    if not _hits:
        _tars = sorted(_glob.glob("/kaggle/input/**/*.tar", recursive=True))
        print("[eb] tars", _tars, flush=True)
        for _t in _tars:
            _d = "/kaggle/working/oai_npz"
            _os.makedirs(_d, exist_ok=True)
            import tarfile as _tf
            with _tf.open(_t) as _f:
                _f.extractall(_d)
            _hits = sorted(_glob.glob(_d + "/**/*.npz", recursive=True))
            if _hits: break
    if _hits:
        _oai = _os.path.dirname(_hits[0])
    else:
        _cand = ["/kaggle/input/rsna-knee-moaks-oai-128"]
        _oai = next((p for p in _cand if _os.path.isdir(p)), _cand[-1])
    print("[iso] oai_dir", _oai, "npz", len(_glob.glob(_oai + "/*.npz")), flush=True)
    import sys as _sys
    _soft = next(iter(sorted(_glob.glob('/kaggle/input/datasets/**/oai_soft_targets.csv', recursive=True))), None)
    print('[iso] soft', _soft, flush=True)
    _sys.argv = [_sys.argv[0]] + ['--ckpt', 'timm', '--arch', 'convnext_small.fb_in22k_ft_in1k', '--res', '384', '--k', '10', '--k_eval', '20', '--epochs', '16', '--bs', '4', '--bb_lr', '2e-5', '--head_lr', '1e-3', '--wd', '0.02', '--workers', '4', '--topk', '3', '--grad_ckpt', '--folds', '0', '--seed', '46', '--tag', 'cnxt_soft46_16e', '--norm', 'imagenet', '--oai_dir', _oai] + (['--oai_soft', _soft] if _soft else [])
    main()
