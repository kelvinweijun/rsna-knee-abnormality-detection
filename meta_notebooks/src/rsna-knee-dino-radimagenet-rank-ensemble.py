# %% [markdown]
# # RSNA Knee DINO–RadImageNet–Raptor Ensemble
# 
# An inference-only, one-dataset reproduction of [Evgen Dvorkin's public RSNA Baseline](https://www.kaggle.com/code/evgendvorkin/rsna-baseline), pinned at version 15. It uses **only the public model assets used by that notebook**.
# 
# The source recipe has a verified **0.941 public LB** result. This consolidated version must be scored separately; **0.942 is a target, not a claimed result**. No training or competition submission runs here.
# 
# ## How many models?
# 
# **41 unique checkpoint members**, across five groups:
# 
# | Group | Unique members | How predictions are combined |
# |---|---:|---|
# | DINOv2-small | 20 | Per-member, per-target percentile ranks, equal mean |
# | DINOv3-small | 5 | Five fold ranks, equal mean; 45% of the transformer blend |
# | RadImageNet attention heads | 10 | Five reference + five E13 heads, sharing one ResNet-50 encoder |
# | Public Raptor CoAtNet | 3 | v5, v10, v8; v5 is reused for a reverse-channel view |
# | Public residual-gated CoAtNet | 3 | Epochs 4, 6, 8 from one training run; equal mean of checkpoint ranks |
# 
# The DINOv2 initialization asset and shared RadImageNet ResNet-50 encoder are included in the dataset. They are not extra ensemble votes. DINOv2 loads each complete competition checkpoint directly into its configured architecture, avoiding redundant initialization-weight loading. A frozen **88-input, 12-output linear calibration table** is also included. Repeated views and epoch snapshots are not independent training folds.
# 
# ## Workflow and blend
# 
# ```text
# Competition DICOMs
#   ├─ 20 DINOv2-small → mean of ranks ─┐
#   ├─  5 DINOv3-small → mean of ranks ─┴─ 55% / 45% transformer blend
#   │                                              │
#   ├─ RadImageNet encoder → reference/E13 heads ────┤
#   │    E13 layout + second layout + flipped view  │
#   │                                  frozen calibration → T
#   │
#   ├─ Raptor v5/v10/v5-reverse/v8 → 60/10/10/20 probability mean → P
#   └─ Residual CoAtNet epochs 4/6/8 → mean of ranks → C
# 
#               H = rank(0.60 × rank(P) + 0.40 × rank(C))
#   final[target] = rank((1 − w[target]) × rank(T) + w[target] × H)
#                                       │
#                                submission.csv
# ```
# 
# `rank` means per-target average percentile rank across **all** test studies. The public Raptor stage first applies the original ordinal rank implementation; the hybrid re-ranks it exactly as the source does. Rank positions and probability means cannot be interchanged.
# 
# | Finding | Raptor hybrid H | DINO/Rad/calibration T |
# |---|---:|---:|
# | ACL, Lateral OA, Fracture | 75% | 25% |
# | Medial Meniscus | 80% | 20% |
# | Lateral Meniscus | 100% | 0% |
# | MCL, Medial OA, PF OA, Effusion, Synovitis, Baker's, Contusion | 60% | 40% |
# 
# Inside **T**, reference and E13 Rad ranks are mixed 50/50. That Rad branch receives a 50% vote on ten findings; Baker's and Fracture retain the transformer values at this step. The same E13 heads are then reused on the second slot layout with normal and horizontal-flip views; this branch receives 15%. Finally, the frozen calibrator adds a 40% rank vote on ACL, Medial OA, Lateral OA, PF OA, Effusion, Baker's and Contusion.
# 
# ## Inputs and reproducibility
# 
# Attach only the competition and **one dataset**: [consolidated assets](https://www.kaggle.com/datasets/tonylica/rsna-knee-bend-dinov3-0917-repro-assets). The notebook verifies the exact manifest and every file, then runs readable Python modules from `pipeline/`. The dataset contains checkpoints, model definitions, calibration, the pinned OpenCV wheel, source provenance and workflow documentation. No private experimental model, report-label table or training cache is needed.
# 
# Preprocessing remains specific to each trained family: DINO crops use 130 mm; Raptor uses its 140 mm/336–384 px settings; the residual CoAtNet uses its own 130 mm physical triplet renderer. These representations are preserved rather than forced into one image cache.
# 
# ## Credits
# 
# Built from the public work of **Evgen Dvorkin, Mattia Angeli, Dread Development / Johnathan Wagner, Pilkwang, Antoine, Sofia Anjenje, Prvsiyan, Marwan and Meta**, as attributed by the source notebook and its asset providers. Exact source references and component licenses are recorded in `SOURCE_AND_LICENSES.md` and `bundle_manifest.json`. This package reproduces checkpoint inference; it does not claim that all upstream models can be retrained bit-for-bit from the material their authors published.
# 


# %% [markdown]
# ## 1. Verify the complete asset bundle
# 
# Every file is checked before loading models. This run requires an offline T4 × 2 session.


# %% cell 2
import hashlib
import json
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '4')
from pathlib import Path
import subprocess
import sys
import time
import pandas as pd

ASSET_SLUG = 'rsna-knee-bend-dinov3-0917-repro-assets'
ASSET = next((p for p in [
    Path('/kaggle/input/datasets/tonylica') / ASSET_SLUG,
    Path('/kaggle/input') / ASSET_SLUG,
] if (p / 'bundle_manifest.json').is_file()), None)
ROOT = next((p for p in [
    Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
    Path('/kaggle/input/rsna-knee-abnormality-detection'),
] if (p / 'test.csv').is_file()), None)
assert ASSET is not None, 'Attach the consolidated reproduction dataset'
assert ROOT is not None and (ROOT / 'test_series').is_dir(), 'Attach the competition data'

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

manifest_path = ASSET / 'bundle_manifest.json'
observed_manifest_sha256 = sha256_file(manifest_path)
assert observed_manifest_sha256 == '3b6279ab50ca8c3ce100644eadda9bc14e0359c05120990e28a43edc6a5364a8', f'Dataset version or manifest mismatch: {observed_manifest_sha256}'
manifest = json.loads(manifest_path.read_text())
for record in manifest['files']:
    path = ASSET / record['path']
    assert path.is_relative_to(ASSET) and path.is_file(), record['path']
    assert path.stat().st_size == record['bytes'], f"Size mismatch: {record['path']}"
    assert sha256_file(path) == record['sha256'], f"Hash mismatch: {record['path']}"
print(f"Verified {manifest['file_count']} files; all models come from public Baseline v15.")
os.environ.update(RSNA_ASSET_ROOT=str(ASSET), RSNA_COMPETITION_ROOT=str(ROOT),
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
WORK = Path('/kaggle/working')
(WORK / 'branches').mkdir(exist_ok=True)
# Remove only a stale primary artifact from a previous interactive run.
(WORK / 'submission.csv').unlink(missing_ok=True)
STAGE_TIMES = {}

def run_stage(script):
    started = time.monotonic()
    subprocess.run([sys.executable, '-u', str(ASSET / 'pipeline' / script)],
                   cwd=WORK, check=True, env=dict(os.environ))
    STAGE_TIMES[script] = round(time.monotonic() - started, 3)
    print(f"Completed {script} in {STAGE_TIMES[script]:.1f}s")
    (WORK / 'stage_times.json').write_text(json.dumps(STAGE_TIMES, indent=2))



# %% [markdown]
# ## 2. DINO and RadImageNet branch
# 
# Runs 20 DINOv2 models, five DINOv3 models, ten Rad heads with their prescribed views, then the frozen calibration.
# 
# Readable implementation: `pipeline/transformer_rad.py` in the attached dataset.


# %% cell 4
run_stage('transformer_rad.py')


# %% [markdown]
# ## 3. Public Raptor branch
# 
# Runs three CoAtNet checkpoints in four views and combines probabilities at 60/10/10/20.
# 
# Readable implementation: `pipeline/public_raptor.py` in the attached dataset.


# %% cell 6
run_stage('public_raptor.py')


# %% [markdown]
# ## 4. Residual CoAtNet branch
# 
# Runs epochs 4, 6 and 8 with the pinned OpenCV renderer, then averages the three per-target rank vectors.
# 
# Readable implementation: `pipeline/residual_coat.py` in the attached dataset.


# %% cell 8
run_stage('residual_coat.py')


# %% [markdown]
# ## 5. Final target-specific ensemble
# 
# Forms the 60/40 Raptor hybrid, applies the target weights above, and validates every submission row.
# 
# Readable implementation: `pipeline/blend.py` in the attached dataset.


# %% cell 10
run_stage('blend.py')


# %% [markdown]
# ## 6. Submission artifact
# 
# After all checks pass, use this notebook version and select `submission.csv` in Kaggle’s **Submit to Competition** flow. This notebook does not submit automatically.


# %% cell 12
submission = pd.read_csv('/kaggle/working/submission.csv', dtype={'StudyInstanceUID': str})
print(f'Ready: {len(submission)} studies, {len(submission.columns)-1} targets')
display(submission.head())
display(json.loads(Path('/kaggle/working/run_receipt.json').read_text()))
