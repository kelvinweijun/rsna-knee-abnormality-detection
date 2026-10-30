import pytest

from tools import publish_folds as pf


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_stage_root_moves_checkpoints_and_drops_strays(tmp_path):
    _touch(tmp_path / "models" / "fold0.pt")
    _touch(tmp_path / "models" / "fold1.pt")
    _touch(tmp_path / "models" / "cv_summary.json")
    _touch(tmp_path / "models" / "notafold.pt")
    _touch(tmp_path / "cache_train.npy")
    files = sorted(p for p in tmp_path.rglob("*") if p.is_file())
    moved = pf.stage_root(files, tmp_path)
    assert {p.name for p in moved} == {"fold0.pt", "fold1.pt",
                                       "cv_summary.json"}
    assert (tmp_path / "fold0.pt").is_file()
    assert (tmp_path / "cv_summary.json").is_file()
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "cache_train.npy").exists()


def test_stage_root_refuses_when_nothing_matches(tmp_path):
    _touch(tmp_path / "models" / "notafold.pt")
    files = [tmp_path / "models" / "notafold.pt"]
    moved = pf.stage_root(files, tmp_path)
    assert moved == []
    with pytest.raises(SystemExit, match="refusing to publish"):
        pf.require_checkpoints(moved)
