import pytest
import torch

from knee_meta import infer
from knee_meta.config import N_LABELS, PREPROC
from knee_meta.model import build_model


def _tiny(img_size=64):
    return build_model(tiny=True, img_size=img_size)


def _mask(n):
    return torch.ones(n, 6)


def test_forward_accepts_unit_range():
    out = _tiny()(torch.rand(2, 6, 3, 64, 64), _mask(2))
    assert out.shape == (2, N_LABELS)
    assert torch.isfinite(out).all()


def test_forward_rejects_uint8():
    imgs = torch.randint(0, 256, (2, 6, 3, 64, 64), dtype=torch.uint8)
    with pytest.raises(ValueError, match="uint8"):
        _tiny()(imgs, _mask(2))


def test_forward_rejects_raw_float_pixels():
    imgs = torch.rand(2, 6, 3, 64, 64) * 255.0
    with pytest.raises(ValueError, match="outside"):
        _tiny()(imgs, _mask(2))


def test_forward_rejects_prenormalized_inputs():
    imgs = torch.full((2, 6, 3, 64, 64), -2.0)
    with pytest.raises(ValueError, match="outside"):
        _tiny()(imgs, _mask(2))


def test_forward_interpolates_larger_cache_to_model_size():
    model = _tiny(img_size=32)
    out = model(torch.rand(1, 6, 3, 64, 64), _mask(1))
    assert out.shape == (1, N_LABELS)


def _save_blob(path, preproc=PREPROC, img_size=64, include=True):
    model = build_model(tiny=True, img_size=img_size)
    blob = {"state_dict": model.state_dict(), "img_size": img_size,
            "val_auc": 0.5, "fold": 0}
    if include:
        blob["preproc"] = preproc
    torch.save(blob, path)


def test_load_models_accepts_matching_preproc(tmp_path):
    _save_blob(tmp_path / "fold0.pt")
    models = infer.load_models(str(tmp_path), torch.device("cpu"), tiny=True)
    assert len(models) == 1
    assert models[0][0] == "fold0.pt"
    assert models[0][2] == 64


def test_load_models_rejects_missing_preproc(tmp_path):
    _save_blob(tmp_path / "fold0.pt", include=False)
    with pytest.raises(ValueError, match="preproc"):
        infer.load_models(str(tmp_path), torch.device("cpu"), tiny=True)


def test_load_models_rejects_wrong_preproc(tmp_path):
    _save_blob(tmp_path / "fold0.pt", preproc="raw255")
    with pytest.raises(ValueError, match="preproc"):
        infer.load_models(str(tmp_path), torch.device("cpu"), tiny=True)
