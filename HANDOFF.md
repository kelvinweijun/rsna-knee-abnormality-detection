# RSNA Knee — OpenCode handoff (2026-10-21 ~04:15 UTC)

Deadline: **2026-10-22 23:59 UTC**. Best LB: **0.958** (do not regress). Submission account: **varianceofx** (submissions only — never training).

## What's running in the cloud (machine-independent)
| Kernel | Account | Purpose | Status |
|---|---|---|---|
| `isotriage/rsna-knee-train-train-full-a` | isotriage | CoAtNet-384, full data, seed 42 | RUNNING |
| `ephemeralbyte/rsna-knee-train-full-b` | ephemeralbyte | CoAtNet-384, full data, seed 777 | RUNNING |
| `ephemeralbyte/rsna-knee-train-train-cnx-t` | ephemeralbyte | ConvNeXt-T-384, full data, seed 2024 | RUNNING |
| submission `57459811` (h9589) | varianceofx | OAI folds w=0.25, hardened guards | PENDING (queued ~01:14 UTC) |

Poll from any machine:
```
set KAGGLE_API_TOKEN=<varianceofx token>   (PowerShell: $env:KAGGLE_API_TOKEN="...")
python poll.py <token_json> <owner/slug>
kaggle competitions submissions rsna-knee-abnormality-detection
```

## Accounts / auth
- Kaggle creds (copy `%USERPROFILE%\.kaggle\*.json`): varianceofx (submissions), isotriage, ephemeralbyte, sixmillionfold (quota exhausted), tensorhydra (quota exhausted).
- Tokens in `%TEMP%\*_token.json` expire in ~12 h — re-mint with `python mint.py <credentials.json> <out.json>`.
- **Gotcha:** an expired `KAGGLE_API_TOKEN` silently falls back to `~/.kaggle/credentials.json` (wrong account). Always verify the pushed URL/username.
- `KAGGLE_CONFIG_DIR` does NOT redirect auth. Use env token or swap `~/.kaggle/credentials.json`.

## Local files that matter (this bundle)
- `meta_notebooks\oai_box\train_jev_oai_box.py` — trainer (`--folds 0` = full data; outputs `raptor_ft_<tag>_swa.pt`, `raptor_gold_<tag>.npz`).
- `meta_notebooks\h9589_vox\` — hardened blend notebook (current best recipe family).
- `meta_notebooks\h957_v5\rsna-knee-h957-v5.ipynb` — the 0.958 source recipe.
- `scripts\` — builders/pollers/scoring (`build_full_kernels.py`, `build_h9589.py`, `oai_gold_blend.py`, `perlabel_gold.py`, `fit_perlabel.py`, `poll.py`, `mint.py`, `read_kdisc.py`, `moaks_join.py`, ...).
- `data\trainlabels\fused_labels.csv` — labels uploaded to 3 private datasets.
- `data\reasonknee\` — MOAKS CSVs (`taxonomymatch_kMRI_SQ_MOAKS_BICL00.csv`, 3,017 knees) + `all_studies.csv`.
- `data\gold\raptor_gold_m1coat384ft_ep0.npz` — gold58 truth used by every scoring harness.
- `data\blend_out\` — h9589/h9587/h9588 outputs incl. `blend_receipt.json`, `raptor_raw.csv`, `submission.csv`.

## Key numbers
- OAI fold gold58: f0 .9057 / f1 .9039 / f2 .8906 / f3 .9062 / 4-fold .9083; nartaa384 .9043; OAI-in-raptor-half blend peak .9134.
- Per-label gold58 (w224, wOAI): ACL(.17,.50) MCL(.12,.50) MedMen(.33,.30) LatMen(.28,.30) MedOA(.12,.25) LatOA(.43,.50) PF(.43,.30) Eff(.28,.65) Syn(.12,.45) Baker(.12,.75) Cont(.17,.40) Frac(.50,.60).
- Labels: fused .9547 gold58; tsuyu122 .9124 (discarded); pseudo-labels rejected (in-sample).
- h9589 cell 8 builds the 384 leg from `raptor_raw.csv` (not `raptor_sub.csv`); per-label OAI weighting requires a cell-8 patch (parked).

## MOAKS plan (new laptop, 236 GB)
1. NDA: user logs in via Edge (varianceofx). Create image download package for **OAI V00 3DESS** (query-download). Full set ~150 GB.
2. `pip install nda-tools` → `downloadcmd -dp <package> -d <dir>` (NDA username/password in keyring).
3. Join: MOAKS `(ID, SIDE)` (1=RIGHT, 2=LEFT) → `all_studies.csv` gives `patient_id/time/study_folder/series_folder/knee_side`.
4. Supervise only findings MOAKS covers: effusion `EFFWK`, synovitis `SYIC`, ACL `ACLTR`, meniscus `MMT*`, BML `MBM*`, osteophytes `MOS*`, cyst `POPCYS`, bursitis `AN/IT/PPBUR`.
5. Expected gain is modest (~+0.001–0.002); do not jeopardize h9590/submission slots.

## Next actions (priority)
1. Poll h9589 score (PENDING).
2. Poll 3 training kernels → pull `raptor_gold_*.npz` + weights; gate on gold58 > 0.906.
3. Build h9590 = best base + accepted full-data arms; submit only if gold58 improves.
4. MOAKS: only if NDA download path works cleanly.

## Copy checklist (new laptop)
1. Install Python 3.11 + `pip install kaggle==1.7.4.5 kagglesdk numpy pandas`.
2. Copy workspace → **same absolute path** `C:\Users\kelvi\Downloads\RSNA Detection` (session path match).
3. Copy `%USERPROFILE%\.kaggle\` (credentials) and this bundle to `%TEMP%\`.
4. Copy opencode state: close opencode here, then copy `%USERPROFILE%\.local\share\opencode\` (`opencode.db`, `-wal`, `-shm`, `auth.json`, `storage\`, `snapshot\`) and `%USERPROFILE%\.config\opencode\`.
5. Verify: `kaggle config view` (username), `python poll.py ...`, reopen opencode in the workspace.
