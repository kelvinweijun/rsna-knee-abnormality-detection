import numpy as np
import pandas as pd
import pytest

from knee_meta import imaging
from knee_meta.config import N_SLOT, SLOTS

try:
    import pydicom
except ImportError:
    pydicom = None

pytestmark = pytest.mark.skipif(pydicom is None, reason="pydicom not installed")

IOP = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def uidify(name):
    import zlib
    return f"1.2.826.1.1.1.{zlib.crc32(str(name).encode('utf-8')) % 100000000}"


def write_slice(path, study, series, sop, ipp, size=128, spacing=1.0):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

    rng = np.random.default_rng(int(ipp[2]) * 7919 + size)
    data = rng.integers(0, 500, (size, size), dtype=np.uint16)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = sop
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = sop
    ds.StudyInstanceUID = uidify(study)
    ds.SeriesInstanceUID = uidify(series)
    ds.Modality = "MR"
    ds.SeriesDescription = "T2 FS SAG"
    ds.Rows = size
    ds.Columns = size
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [spacing, spacing]
    ds.SliceThickness = spacing
    ds.ImagePositionPatient = [float(x) for x in ipp]
    ds.ImageOrientationPatient = IOP
    ds.PixelData = data.tobytes()
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    pydicom.dcmwrite(str(path), ds, enforce_file_format=True)
    return data


def make_series(root, split, study, series, plane, fluid, n_slices=20,
                origin_x=-100.0, shuffle_seed=0):
    sdir = root / split / study / series
    sdir.mkdir(parents=True, exist_ok=True)
    order = np.random.default_rng(shuffle_seed).permutation(n_slices)
    for i in order:
        ipp = [origin_x, 0.0, float(i) * 1.0]
        write_slice(sdir / f"{i:04d}.dcm", study, series,
                    uidify(f"{study}.{series}.{i}"), ipp)
    return {
        "StudyInstanceUID": study,
        "SeriesInstanceUID": series,
        "Fluid_Sensitive": fluid,
        "Fat_Suppression": fluid,
        "Anatomical_Plane": plane,
    }


@pytest.fixture()
def corpus(tmp_path):
    root = tmp_path / "comp"
    rows = [
        make_series(root, "train_series", "st1", "se_sag_r", "Sagittal", 1,
                    shuffle_seed=1),
        make_series(root, "train_series", "st1", "se_cor_r", "Coronal", 1,
                    shuffle_seed=2),
        make_series(root, "train_series", "st2", "se_sag_l", "Sagittal", 1,
                    origin_x=-40.0, shuffle_seed=3),
        make_series(root, "train_series", "st3", "se_ax_struct", "Axial", 0,
                    shuffle_seed=4),
    ]
    pd.DataFrame(rows).to_csv(root / "train_series.csv", index=False)
    pd.DataFrame(rows[:1]).to_csv(root / "test_series.csv", index=False)
    return root


def test_slot_index_matches_config():
    assert imaging.slot_index("Sagittal", 1) == 0
    assert imaging.slot_index("Sagittal", 0) == 1
    assert imaging.slot_index("Coronal", 1) == 2
    assert imaging.slot_index("Axial", 0) == 5
    assert N_SLOT == len(SLOTS) == 6


def test_order_slices_uses_geometry(tmp_path):
    root = tmp_path / "c"
    make_series(root, "train_series", "st1", "se1", "Sagittal", 1, n_slices=12)
    files = imaging.list_files(root / "train_series" / "st1" / "se1", "se1")
    headers = imaging.read_headers(files)
    order = imaging.order_slices(headers)
    zs = []
    for i in order:
        ipp = imaging._ipp_vec(headers[i])
        zs.append(ipp[2])
    assert zs == sorted(zs)


def test_laterality_from_geometry(corpus):
    files = imaging.list_files(corpus / "train_series" / "st1" / "se_sag_r",
                               "se_sag_r")
    headers = imaging.read_headers(files)
    assert imaging.laterality(headers) == "R"
    files = imaging.list_files(corpus / "train_series" / "st2" / "se_sag_l",
                               "se_sag_l")
    headers = imaging.read_headers(files)
    assert imaging.laterality(headers) == "L"


def test_crop_resize_keeps_output_square():
    arr = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128)
    out = imaging.crop_resize(arr, 1.0, 1.0, 64, crop_mm=130.0)
    assert out.shape == (64, 64)
    wide = imaging.crop_resize(arr, 0.5, 2.0, 64, crop_mm=130.0)
    assert wide.shape == (64, 64)


def test_window_to_uint8():
    arr = np.linspace(-1000, 1000, 1000, dtype=np.float32)
    out = imaging.window_to_uint8(arr, 0.0, 400.0)
    assert out.dtype == np.uint8
    assert out[0] == 0 and out[-1] == 255
    fallback = imaging.window_to_uint8(arr, None, None)
    assert fallback.min() == 0 and fallback.max() == 255


def test_build_cache_masks_and_flip(corpus):
    cache, mask, studies = imaging.build_cache(str(corpus), "train",
                                               img_size=64, threads=4)
    assert cache.shape == (3, N_SLOT, 3, 64, 64)
    assert studies.tolist() == ["st1", "st2", "st3"]
    assert mask[0].tolist() == [1, 0, 1, 0, 0, 0]
    assert mask[1].tolist() == [1, 0, 0, 0, 0, 0]
    assert mask[2].tolist() == [0, 0, 0, 0, 0, 1]
    right = cache[0, 0]
    left = cache[1, 0]
    assert right.max() > 0
    assert np.array_equal(left, right[:, :, ::-1])


def test_cache_round_trip(corpus, tmp_path):
    cache, mask, studies = imaging.build_cache(str(corpus), "train",
                                               img_size=64, threads=4)
    path = str(tmp_path / "cache.npy")
    imaging.save_cache(path, cache, mask, studies)
    cache2, mask2, studies2 = imaging.load_cache(path)
    np.testing.assert_array_equal(cache, cache2)
    np.testing.assert_array_equal(mask, mask2)
    assert studies2.tolist() == studies.tolist()


def test_sample_indices_inside_band():
    idx = imaging._sample_indices(20, (0.2, 0.8), 3)
    assert idx == sorted(idx)
    assert all(0 <= i < 20 for i in idx)
    assert idx[0] >= 3 and idx[-1] <= 16
