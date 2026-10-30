"""Five-fold CV training: confidence-weighted BCE plus a pairwise ranking term."""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold

from .config import (BATCH_STUDIES, EPOCHS, EVAL_BATCH, GOLD_WEIGHT, LR_BACKBONE,
                     LR_HEAD, N_FOLDS, PAIRWISE_W, PREPROC, SEED, TIME_BUDGET_S,
                     TRAIN_IMG_SIZE, UNFREEZE_LAST, WEIGHT_DECAY, find_root)
from .dataset import make_dataset, make_loader
from .imaging import load_cache
from .labels import (apply_consensus, audit, load_label_table,
                     resolve_consensus, save_label_table)
from .metrics import macro_auc, per_label_auc
from .model import build_model, param_groups


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def array_stats(arr, target=2_000_000):
    """Mean and std from a strided sample, never a full-array float64 copy."""
    flat = arr.reshape(-1)
    step = max(1, flat.size // target)
    sample = flat[::step]
    return float(sample.mean()), float(sample.std())


def average_states(states):
    """Mean of cloned state dicts; non-float buffers keep the last snapshot."""
    out = {}
    for key in states[0]:
        vals = [s[key] for s in states]
        if torch.is_floating_point(vals[0]):
            out[key] = torch.stack(vals).float().mean(0).to(vals[0].dtype)
        else:
            out[key] = vals[-1].clone()
    return out


def parse_fold_spec(spec):
    """'0' or '0,1,2' -> set of fold ids; None/'' stays None."""
    if spec is None or not str(spec).strip():
        return None
    return {int(x) for x in str(spec).split(",") if str(x).strip()}


def enable_determinism():
    """Strict bitwise-reproducible kernels (CUDA); must run before matmuls."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def determinism_check(model, batch, device, img_size, seed=1234):
    """Two identical forward/backward passes -> (ok, why, strict_unusable)."""
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = (torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None)

    def run_pass():
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model.train()
        try:
            _, _, _, _, loss = step(model, batch, device, img_size, train=True)
            loss.backward()
        except RuntimeError as exc:
            model.zero_grad(set_to_none=True)
            return None, str(exc)
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                 if p.grad is not None}
        model.zero_grad(set_to_none=True)
        return grads, None

    g1, e1 = run_pass()
    g2, e2 = run_pass()
    model.load_state_dict(state)
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    if e1 is not None or e2 is not None:
        return False, (e1 or e2)[:300], True
    same = all(torch.allclose(g1[k], g2[k], rtol=0.0, atol=0.0, equal_nan=True)
               for k in g1)
    return same, ("bitwise equal" if same else "gradients differ"), False


def weighted_bce(logits, target, conf, gold):
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if gold.dim() == 1:
        gold = gold.unsqueeze(1)
    w = conf * (1.0 + (GOLD_WEIGHT - 1.0) * gold)
    return (per * w).sum() / w.sum().clamp_min(1e-6)


def pairwise_loss(logits, target, conf, margin=1.0):
    if PAIRWISE_W <= 0:
        return logits.sum() * 0.0
    sign = target[:, None, :] - target[None, :, :]
    valid = sign != 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    gap = logits[:, None, :] - logits[None, :, :]
    w = torch.minimum(conf[:, None, :], conf[None, :, :])
    loss = F.relu(margin - torch.sign(sign) * gap)
    return (loss * valid.float() * w).sum() / (valid.float() * w).sum().clamp_min(1e-6)


def step(model, batch, device, img_size, train=True):
    imgs = batch["imgs"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    conf = batch["conf"].to(device, non_blocking=True)
    gold = batch["gold"].to(device, non_blocking=True)
    logits = model(imgs, mask, img_size=img_size)
    loss = weighted_bce(logits, target, conf, gold)
    if train:
        loss = loss + PAIRWISE_W * pairwise_loss(logits, target, conf)
    return logits.detach(), target, conf, gold, loss


@torch.no_grad()
def evaluate(model, loader, device, img_size):
    model.eval()
    preds, truths = [], []
    total = 0.0
    n = 0
    for batch in loader:
        logits, target, conf, gold, loss = step(model, batch, device, img_size,
                                                train=False)
        preds.append(logits.cpu().numpy())
        truths.append(target.cpu().numpy())
        total += float(loss) * len(target)
        n += len(target)
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return macro_auc(truths, preds), total / max(n, 1), per_label_auc(truths, preds)


def train_fold(fold, tr_idx, va_idx, studies, cache, mask, table, args, device):
    train_ds = make_dataset(cache, mask, studies, table, train=True,
                            study_ids=[studies[i] for i in tr_idx],
                            img_size=args.img_size)
    val_ds = make_dataset(cache, mask, studies, table, train=False,
                          study_ids=[studies[i] for i in va_idx],
                          img_size=args.img_size)
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, args.eval_batch, shuffle=False)
    model = build_model(unfreeze_last=args.unfreeze, source=args.backbone,
                        img_size=args.img_size, tiny=getattr(args, "tiny", False))
    model.to(device)
    groups, extra = param_groups(model, LR_HEAD, LR_BACKBONE, WEIGHT_DECAY)
    opt = torch.optim.AdamW(groups, **extra)
    if getattr(args, "deterministic", False):
        rng_cpu = torch.get_rng_state()
        rng_cuda = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        probe = next(iter(train_loader))
        ok, why, strict_unusable = determinism_check(
            model, probe, device, args.img_size)
        print(f"determinism check: {'PASS' if ok else 'FAIL'} ({why})",
              flush=True)
        if strict_unusable:
            torch.use_deterministic_algorithms(False)
            ok, why, _ = determinism_check(model, probe, device, args.img_size)
            print(f"determinism check (best-effort): "
                  f"{'PASS' if ok else 'FAIL'} ({why})", flush=True)
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)
    total_steps = max(1, args.epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)) *
        (0.5 * (1.0 + np.cos(np.pi * min(s, total_steps) / total_steps))))
    best, best_state, history = -1.0, None, []
    swa_states = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            _, _, _, _, loss = step(model, batch, device, args.img_size, train=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(batch["target"])
            n += len(batch["target"])
        val_auc, val_loss, per_label = evaluate(model, val_loader, device,
                                                args.img_size)
        history.append({"epoch": epoch, "train_loss": total / max(n, 1),
                        "val_auc": val_auc, "val_loss": val_loss,
                        "per_label": per_label})
        print(f"fold {fold} epoch {epoch}: train {total / max(n, 1):.4f} "
              f"val_auc {val_auc:.4f}", flush=True)
        if val_auc > best:
            best = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if args.swa_from >= 0 and epoch >= args.swa_from:
            swa_states.append({k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()})
        if time.time() - started > args.budget_per_fold:
            print(f"fold {fold}: budget reached at epoch {epoch}", flush=True)
            break
    swa = {}
    if swa_states:
        swa_state = average_states(swa_states)
        model.load_state_dict(swa_state)
        swa_auc, swa_loss, _ = evaluate(model, val_loader, device, args.img_size)
        swa = {"swa_val_auc": swa_auc, "swa_members": len(swa_states),
               "swa_from": args.swa_from, "swa_selected": swa_auc > best}
        print(f"fold {fold} swa: {swa_auc:.4f} "
              f"({len(swa_states)} members, selected={swa_auc > best})", flush=True)
        if swa_auc > best:
            best = swa_auc
            best_state = swa_state
    elif args.swa_from >= 0:
        print(f"fold {fold}: no swa members (swa_from={args.swa_from})", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    out = Path(args.out) / f"fold{fold}.pt"
    torch.save({"state_dict": model.state_dict(), "val_auc": best,
                "img_size": args.img_size, "fold": fold, "preproc": PREPROC,
                "history": history, **swa}, out)
    return best, history


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default="models")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_STUDIES)
    ap.add_argument("--eval-batch", type=int, default=EVAL_BATCH)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--unfreeze", type=int, default=UNFREEZE_LAST)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--budget", type=float, default=TIME_BUDGET_S)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--only-fold", type=str, default=None,
                    help="train only these folds, '0' or '0,1,2' (split "
                         "sessions for long runs)")
    ap.add_argument("--swa-from", type=int, default=-1,
                    help="average weights from this epoch onward; -1 disables")
    ap.add_argument("--deterministic", action="store_true",
                    help="reproducible kernels (paired screening)")
    ap.add_argument("--consensus", default=None,
                    help="path or 'auto': downweight cells where a second "
                         "labeller disagrees (values unchanged)")
    args = ap.parse_args(argv)
    args.only_fold = parse_fold_spec(args.only_fold)
    args.budget_per_fold = args.budget / max(1, args.folds)
    if args.deterministic:
        enable_determinism()
    seed_everything(args.seed)

    root = Path(args.root) if args.root else find_root()
    cache_path = Path(args.cache) if args.cache else root / "cache_train.npy"
    label_path = Path(args.labels) if args.labels else Path("report_labels.csv")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if label_path.is_file():
        table = load_label_table(label_path)
    else:
        table = save_label_table(root / "train.csv", label_path)
    print("label audit:", json.dumps(audit(table, root / "train.csv"), indent=2))
    if args.consensus:
        table, cons_stats = apply_consensus(
            table, resolve_consensus(args.consensus))
        print("consensus:", json.dumps(cons_stats), flush=True)

    cache, mask, studies = load_cache(str(cache_path))
    studies = [str(s) for s in studies]
    known = set(table.StudyInstanceUID)
    keep = [i for i, s in enumerate(studies) if s in known]
    if len(keep) != len(studies):
        cache, mask, studies = cache[keep], mask[keep], [studies[i] for i in keep]
    if args.img_size is None:
        args.img_size = TRAIN_IMG_SIZE
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "cpu"))
    print(f"cache {tuple(cache.shape)} device {device} img {args.img_size}",
          flush=True)
    print("slot fill:", np.round(mask.mean(axis=0), 3).tolist(),
          "| pixel mean %.1f std %.1f" % array_stats(cache), flush=True)

    order = np.random.RandomState(args.seed).permutation(len(studies))
    splits = KFold(n_splits=args.folds, shuffle=False)
    results = []
    started = time.time()
    for fold, (tr, va) in enumerate(splits.split(order)):
        if args.only_fold is not None and fold not in args.only_fold:
            continue
        auc, history = train_fold(fold, order[tr], order[va], studies, cache,
                                  mask, table, args, device)
        results.append({"fold": fold, "val_auc": auc})
        print(f"fold {fold} best val_auc {auc:.4f}", flush=True)
        if time.time() - started > args.budget:
            print("global budget reached", flush=True)
            break
    summary = Path(args.out) / "cv_summary.json"
    summary.write_text(json.dumps(results, indent=2))
    print("cv:", json.dumps(results), flush=True)
    return results


if __name__ == "__main__":
    main()
