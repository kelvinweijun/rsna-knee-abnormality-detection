import pandas as pd
import pytest

from knee_meta import labels
from knee_meta.config import TARGETS

CASES = [
    ("Complete tear of the anterior cruciate ligament. Small joint effusion.",
     {"ACL": 1, "Effusion": 1}),
    ("No tear of the ACL. No fracture is seen.",
     {"ACL": 0, "Fracture": 0}),
    ("Rotura de menisco interno. Artrosis femorotibial medial. Derrame.",
     {"Medial Meniscus": 1, "Medial OA": 1, "Effusion": 1}),
    ("Baker cyst of 2 cm. Lateral meniscus tear.",
     {"Baker's": 1, "Lateral Meniscus": 1}),
    ("Normal study.",
     {lab: 0 for lab in TARGETS}),
]


@pytest.mark.parametrize("report,expected", CASES)
def test_extract_labels(report, expected):
    got = labels.extract_labels(report)
    for lab, value in expected.items():
        assert got[lab][0] == value, (report, lab, got[lab])
    for lab in TARGETS:
        assert 0.0 <= got[lab][1] <= 1.0


def test_missing_mentions_carry_low_confidence():
    got = labels.extract_labels("Normal knee study.")
    assert got["ACL"] == (0.0, labels.MISSING_CONF)
    assert got["Effusion"] == (0.0, labels.MISSING_CONF)


def test_negated_mention_is_confident_zero():
    got = labels.extract_labels("No effusion and no synovitis.")
    assert got["Effusion"] == (0.0, labels.MENTION_CONF)
    assert got["Synovitis"] == (0.0, labels.MENTION_CONF)


def test_positive_mentions_are_confident():
    got = labels.extract_labels("Fracture of the tibial plateau.")
    assert got["Fracture"] == (1.0, labels.POSITIVE_CONF)


def test_spanish_report():
    got = labels.extract_labels(
        "Rotura del ligamento cruzado anterior. Derrame articular moderado.")
    assert got["ACL"][0] == 1
    assert got["Effusion"][0] == 1


def test_trauma_clause_does_not_trigger_compartment_oa():
    got = labels.extract_labels(
        "Subchondral fracture of the lateral femoral condyle with bone "
        "contusion and marrow edema.")
    assert got["Lateral OA"][0] == 0.0
    assert got["Fracture"][0] == 1.0
    assert got["Contusion"][0] == 1.0


def test_degenerative_clause_triggers_compartment_oa():
    got = labels.extract_labels("Osteoarthritis of the lateral compartment.")
    assert got["Lateral OA"][0] == 1.0


def test_patellar_facet_context_is_pf_not_compartment():
    got = labels.extract_labels("Chondrosis at the lateral patellar facet.")
    assert got["PF OA"][0] == 1.0
    assert got["Lateral OA"][0] == 0.0


def test_audit_floor_on_gold():
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    audit = labels.audit(None, str(repo / "train.csv"))
    assert audit["n_gold"] == 58
    assert audit["macro_agreement"] >= 0.80


def test_build_label_table(tmp_path):
    train = pd.DataFrame({
        "StudyInstanceUID": ["a", "b"],
        "Report": ["Fracture.", "No fracture seen."],
        **{lab: [float("nan"), float("nan")] for lab in TARGETS},
    })
    path = tmp_path / "train.csv"
    train.to_csv(path, index=False)
    table = labels.build_label_table(str(path))
    assert list(table.columns) == ["StudyInstanceUID", "is_gold"] + [
        c for lab in TARGETS for c in (lab, lab + "__conf")]
    assert table.is_gold.sum() == 0
    row = table.set_index("StudyInstanceUID").loc["a"]
    assert row["Fracture"] == 1.0
    row = table.set_index("StudyInstanceUID").loc["b"]
    assert row["Fracture"] == 0.0
    assert row["Fracture__conf"] == labels.MENTION_CONF
    audit = labels.audit(table, str(path))
    assert audit["n_gold"] == 0


def _mini_table():
    rows = []
    for i, uid in enumerate(["a", "b", "c"]):
        row = {"StudyInstanceUID": uid, "is_gold": 1 if i == 0 else 0}
        for lab in TARGETS:
            row[lab] = 1.0 if i % 2 == 0 else 0.0
            row[lab + "__conf"] = 1.0 if i == 0 else labels.POSITIVE_CONF
        rows.append(row)
    return pd.DataFrame(rows, columns=["StudyInstanceUID", "is_gold"] +
                        [c for lab in TARGETS for c in (lab, lab + "__conf")])


def test_apply_consensus_downweights_only_nongold_disagreements(tmp_path):
    table = _mini_table()
    partner = table[["StudyInstanceUID"] + TARGETS].copy()
    partner.loc[partner.StudyInstanceUID == "a", "ACL"] = 0.0  # gold: skip
    partner.loc[partner.StudyInstanceUID == "b", "ACL"] = 1.0  # flip
    path = tmp_path / "partner.csv"
    partner.to_csv(path, index=False)

    out, stats = labels.apply_consensus(table, str(path))
    assert (out[TARGETS] == table[TARGETS]).all().all()
    idx = out.set_index("StudyInstanceUID")
    assert idx.loc["a", "ACL__conf"] == 1.0
    assert idx.loc["b", "ACL__conf"] == labels.MISSING_CONF
    assert idx.loc["c", "ACL__conf"] == labels.POSITIVE_CONF
    assert stats["n_gold_skipped"] == 1
    assert stats["per_label"]["ACL"] == 1
    assert stats["n_down"] == 1
    assert stats["partner"] == "partner.csv"


def test_resolve_consensus(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        labels.resolve_consensus(str(tmp_path / "nope.csv"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / labels.CONSENSUS_FILE).write_text("StudyInstanceUID\n")
    found = labels.resolve_consensus("auto")
    assert found.name == labels.CONSENSUS_FILE
    assert found.parent == tmp_path
