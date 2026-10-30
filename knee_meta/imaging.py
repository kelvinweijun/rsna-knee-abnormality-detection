"""DICOM to uint8 cache: physical crop, canonical laterality, slot packing."""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import (CACHE_IMG, HDR_THREADS, LAT_MIN_OFFSET_MM, N_SLOT,
                     PIX_THREADS, PLANE_BAND, SLICE_BAND,
                     SLICES_PER_SLOT, SLOTS, CROP_MM)

try:
    import pydicom
except ImportError:
    pydicom = None


def _require_pydicom():
    if pydicom is None:
        raise ImportError("pydicom is required to read the DICOM corpus")


_SERIES_INDEX = {}


def _build_index(root, split):
    key = (str(root), split)
    if key in _SERIES_INDEX:
        return _SERIES_INDEX[key]
    idx = {}
    base = Path(root) / split
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            idx.setdefault(child.name, child)
            for grand in sorted(child.iterdir()):
                if grand.is_dir():
                    idx.setdefault(grand.name, grand)
    _SERIES_INDEX[key] = idx
    return idx


def find_series_dir(root, uid):
    for split in ("train_series", "test_series"):
        hit = _build_index(root, split).get(uid)
        if hit is not None and hit.exists():
            return hit
    return None


def list_files(series_dir, uid):
    if series_dir is None:
        return []
    if series_dir.is_file():
        return [series_dir]
    files = [p for p in series_dir.rglob("*") if p.is_file()]
    files.sort(key=lambda p: p.name)
    return files


def read_headers(files, threads=HDR_THREADS):
    _require_pydicom()

    def one(path):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            return None
        return ds

    with ThreadPoolExecutor(max_workers=threads) as ex:
        out = list(ex.map(one, files))
    return [h for h in out if h is not None]


def _float_seq(value, minimum):
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            out = [float(x) for x in value]
            return out if len(out) >= minimum else None
        except (TypeError, ValueError):
            pass
    text = str(value).strip().strip("[](){}")
    for ch in "\\|,;":
        text = text.replace(ch, " ")
    try:
        out = [float(p) for p in text.split()]
    except ValueError:
        return None
    return out if len(out) >= minimum else None


def _ipp_vec(h, n=3):
    try:
        v = _float_seq(h.ImagePositionPatient, n)
    except Exception:
        return None
    return np.array(v) if v is not None and len(v) >= n else None


def _iop_vec(h, n=6):
    try:
        v = _float_seq(h.ImageOrientationPatient, n)
    except Exception:
        return None
    return np.array(v) if v is not None and len(v) >= n else None


def _pixel_spacing(h):
    try:
        return _float_seq(h.PixelSpacing, 2)
    except Exception:
        return None


def order_slices(headers):
    keys = []
    for i, h in enumerate(headers):
        ipp, iop = _ipp_vec(h), _iop_vec(h)
        if ipp is None or iop is None:
            keys.append((np.inf, i))
            continue
        normal = np.cross(iop[:3], iop[3:6])
        keys.append((float(np.dot(normal, ipp)), i))
    finite = [k for k in keys if np.isfinite(k[0])]
    if len(finite) < max(1, len(keys) // 2):
        return list(range(len(headers)))
    return [i for _, i in sorted(keys, key=lambda t: (t[0], t[1]))]


def laterality(headers):
    cx = []
    for h in headers:
        ipp, iop, ps = _ipp_vec(h), _iop_vec(h), _pixel_spacing(h)
        if ipp is None or iop is None or ps is None:
            continue
        try:
            rows, cols = int(h.Rows), int(h.Columns)
        except Exception:
            continue
        c = ipp + iop[:3] * ps[1] * cols / 2 + iop[3:6] * ps[0] * rows / 2
        cx.append(float(c[0]))
    if not cx:
        return None
    med = float(np.median(cx))
    if abs(med) < LAT_MIN_OFFSET_MM:
        return None
    return "R" if med < 0 else "L"


def crop_resize(arr, ps_rows, ps_cols, out_size, crop_mm=CROP_MM):
    h, w = arr.shape[:2]
    fh, fw = h * ps_rows, w * ps_cols
    if fh >= crop_mm and fw >= crop_mm:
        ch = min(h, int(round(crop_mm / ps_rows)))
        cw = min(w, int(round(crop_mm / ps_cols)))
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        arr = arr[y0:y0 + ch, x0:x0 + cw]
        fh, fw = ch * ps_rows, cw * ps_cols
    scale = out_size / max(fh, fw)
    nh, nw = max(1, int(round(fh * scale))), max(1, int(round(fw * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(arr, (nw, nh), interpolation=interp)
    canvas = np.full((out_size, out_size), int(np.median(resized)), dtype=resized.dtype)
    y0 = (out_size - nh) // 2
    x0 = (out_size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def normalise_window(arr, h):
    slope = float(getattr(h, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(h, "RescaleIntercept", 0.0) or 0.0)
    return arr.astype(np.float32) * slope + inter


def _window(h):
    wc = getattr(h, "WindowCenter", None)
    ww = getattr(h, "WindowWidth", None)
    try:
        return float(wc), float(ww)
    except (TypeError, ValueError):
        return None, None


def window_to_uint8(arr, centre=None, width=None):
    if centre is None or width is None or not width:
        lo, hi = np.percentile(arr, [1.0, 99.0])
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max() + 1.0)
        centre, width = (lo + hi) / 2.0, max(hi - lo, 1.0)
    lo, hi = centre - width / 2.0, centre + width / 2.0
    scaled = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (scaled * 255.0 + 0.5).astype(np.uint8)


def slot_index(plane, fluid):
    for i, (p, fs) in enumerate(SLOTS):
        if p == plane and fs == int(fluid):
            return i
    return -1


def _sample_indices(n, band, count):
    if n <= 0:
        return []
    if count <= 1:
        return [int(round((band[0] + band[1]) / 2 * (n - 1)))]
    lo = band[0] * (n - 1)
    hi = band[1] * (n - 1)
    step = (hi - lo) / (count - 1) if count > 1 else 0.0
    idx = [int(round(lo + i * step)) for i in range(count)]
    return [min(max(i, 0), n - 1) for i in idx]


def build_cache(root, split, img_size=CACHE_IMG, slices_per_slot=SLICES_PER_SLOT,
                threads=PIX_THREADS, progress=None, max_studies=None):
    _require_pydicom()
    root = str(root)
    series_csv = os.path.join(root, f"{split}_series.csv")
    table = pd.read_csv(series_csv, dtype={"StudyInstanceUID": str,
                                           "SeriesInstanceUID": str})
    studies = sorted(table.StudyInstanceUID.unique())
    if max_studies:
        studies = studies[:max_studies]
    index = {uid: i for i, uid in enumerate(studies)}
    cache = np.zeros((len(studies), N_SLOT, slices_per_slot, img_size, img_size),
                     dtype=np.uint8)
    mask = np.zeros((len(studies), N_SLOT), dtype=np.uint8)

    def decode_one(series_row):
        uid = series_row.StudyInstanceUID
        if uid not in index:
            return None
        sdir = find_series_dir(root, series_row.SeriesInstanceUID)
        files = list_files(sdir, series_row.SeriesInstanceUID)
        if not files:
            return None
        headers = read_headers(files)
        if not headers:
            return None
        order = order_slices(headers)
        headers = [headers[i] for i in order]
        files = [files[i] for i in order]
        plane = str(series_row.Anatomical_Plane)
        fluid = int(series_row.Fluid_Sensitive)
        sidx = slot_index(plane, fluid)
        if sidx < 0:
            return None
        band = PLANE_BAND.get(plane, SLICE_BAND)
        want = _sample_indices(len(headers), band, slices_per_slot + 2)
        if len(headers) > slices_per_slot + 2:
            want = want[1:-1][:slices_per_slot]
        side = laterality(headers)
        slices = []
        for k in want:
            h = headers[k]
            try:
                ds = pydicom.dcmread(str(files[k]), force=True)
                arr = normalise_window(ds.pixel_array, h)
            except Exception:
                arr = None
            if arr is None:
                slices = []
                break
            ps = _pixel_spacing(h)
            if ps is None:
                slices = []
                break
            arr = crop_resize(arr, ps[0], ps[1], img_size)
            wc, ww = _window(h)
            arr = window_to_uint8(arr, wc, ww)
            if side == "L":
                arr = np.ascontiguousarray(arr[:, ::-1])
            slices.append(arr)
        if len(slices) != slices_per_slot:
            return None
        return index[uid], sidx, np.stack(slices)

    rows = [r for r in table.itertuples(index=False)
            if r.StudyInstanceUID in index]
    done = 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for result in ex.map(decode_one, rows):
            if result is not None:
                i, sidx, arr = result
                cache[i, sidx] = arr
                mask[i, sidx] = 1
            done += 1
            if progress and done % 200 == 0:
                progress(done, len(rows))
    return cache, mask, np.array(studies, dtype=object)


def save_cache(path, cache, mask, studies):
    np.save(path, cache)
    np.savez(path + ".mask.npz", mask=mask, studies=studies.astype(str))


def load_cache(path):
    cache = np.load(path, allow_pickle=False)
    meta = np.load(path + ".mask.npz", allow_pickle=True)
    return cache, meta["mask"], meta["studies"]
