"""Multilingual rule extraction of the twelve findings from free-text reports.

Every label comes back as (value, confidence). A report that never mentions a
finding is an unknown, not a negative, so it carries a low confidence and pulls
weakly on that output rather than asserting absence.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (LABEL_FILE, MENTION_CONF, MISSING_CONF, N_LABELS,
                     POSITIVE_CONF, TARGETS)

_CHAR_MAP = str.maketrans({"ı": "i", "İ": "i", "ß": "ss", "ø": "o", "Ø": "o",
                           "đ": "d", "Đ": "d"})

TEAR = [
    "tear", "torn", "tearing", "rupture", "ruptured", "disruption", "discontinuity",
    "rotura", "ruptura", "desgarro", "roto", "rota",
    "dechirure", "dechire", "lesion meniscale",
    "scheur", "ruptuur", "gescheurd",
    "riss", "rissbildung", "ruptur", "zerreissung", "einriss",
    "yirtik", "yirtig", "kopma", "butunluk kaybi",
    "ρηξη", "ρηξις", "ρηγμα",
    "руптура", "разкъсв", "разрив", "puknuce", "prekid",
]
SPRAIN = [
    "sprain", "esguince", "entorse", "verstauchung", "verstuiking",
    "distorsiyon", "burkulma", "διαστρεμμα", "навяхван",
    "injury", "lesion", "letsel", "verletzung", "zedelenme",
]
OA_FIND = [
    "osteoarthritis", "osteoarthrosis", "arthrosis", "arthritic",
    "artrosis", "artrose", "arthrose", "gonartrose", "gonartroz", "gonarthrose",
    "gonartro", "artroz",
    "chondrosis", "chondral", "chondropathy", "chondropathie", "chondropatie",
    "chondromalacia", "kondromalazi", "condropatia", "condral", "chondropatia",
    "kraakbeenlijden", "kraakbeenverlies", "kraakbeenschade",
    "knorpelschaden", "knorpeldefekt", "knorpelverlust", "knorpelbelag",
    "cartilage loss", "cartilage thinning", "cartilage fissuring",
    "cartilage defect", "cartilage fissure", "chondral defect", "chondral loss",
    "osteophyt", "osteofit", "osteofyt", "osteophyte", "spurring", "osteofyte",
    "joint space narrowing", "gelenkspaltverschmalerung",
    "kikirdak kaybi", "kikirdak incelme", "kondral",
    "ulcera condral", "hrskavice", "denudacija",
    "χονδροπαθ", "οστεοαρθρ", "οστεοφυτ", "αρθριτ",
    "артроз", "хондропат", "остеофит", "хрущялн",
]
MEDIAL_Q = ["medial", "interno", "interna", "interne", "mediaal", "mediale",
            "innen", "medyal", "medijaln", "εσω", "медиал", "вътреш"]
LATERAL_Q = ["lateral", "externo", "externa", "externe", "buiten", "aussen",
             "lateraln", "εξω", "латерал", "външ"]
PF_Q = ["patellofemoral", "patelofemoral", "femoropatellar", "femoropatelar",
        "femoropatellair", "retropatellar", "retrorotulian", "patellar facet",
        "trochlea", "troclea", "trochlee", "rotula", "rotulian", "patella",
        "patellaire", "patellar", "diz kapagi", "patelofemoraln",
        "επιγονατιδ", "τροχιλ", "пател", "ретропател"]
TRICOMP = ["tricompartmental", "tricompartimental", "three compartments",
           "three compartmens", "all compartments", "gonartrose", "gonartroz",
           "gonarthrose", "gonartrosis", "gonartro", "pangonartro"]

SPECIFIC = {
    "ACL": ["acl", "anterior cruciate", "lca", "ligamento cruzado anterior",
            "ligament croise anterieur", "croise anterieur",
            "voorste kruisband", "vkb", "vorderes kreuzband", "vorderen kreuzband",
            "vordere kreuzband", "on capraz bag", "anterior capraz bag",
            "προσθιο χιαστ", "προσθιου χιαστ", "προσθιος χιαστ",
            "предна кръстна", "предната кръстна", "prednji ukrizeni"],
    "MCL": ["mcl", "medial collateral", "lcm", "ligamento colateral medial",
            "ligamento colateral interno", "ligament collateral medial",
            "collateral medial", "mediale collaterale", "mediaal collateraal",
            "innenband", "mediales kollateralband", "medialen kollateralband",
            "mediale kollateralband", "medyal kollateral", "ic yan bag",
            "εσω πλαγιο", "медиален колатерален", "вътрешна колатерална"],
    "Medial Meniscus": [
        "medial meniscus", "meniscus medialis", "menisco medial", "menisco interno",
        "mediale meniscus", "binnenmeniscus", "innenmeniskus", "meniskus medialis",
        "menisque interne", "menisque medial", "medial menisk", "medyal menisk",
        "εσω μηνισκ", "медиалния менискус", "медиален мениск", "вътрешния мениск",
        "medial and lateral menisc", "medial ve lateral menisk",
        "menisco medial y lateral", "menisco interno y externo",
        "menisco interno y lateral", "mediale en laterale meniscus",
        "innen und aussenmeniskus", "medijalnog meniskusa"],
    "Lateral Meniscus": [
        "lateral meniscus", "meniscus lateralis", "menisco lateral",
        "menisco externo", "laterale meniscus", "buitenmeniscus",
        "aussenmeniskus", "meniskus lateralis", "menisque externe",
        "menisque lateral", "lateral menisk", "εξω μηνισκ",
        "латералния менискус", "латерален мениск", "външния мениск",
        "medial and lateral menisc", "medial ve lateral menisk",
        "menisco medial y lateral", "menisco interno y externo",
        "menisco interno y lateral", "mediale en laterale meniscus",
        "innen und aussenmeniskus", "lateralnog meniskusa"],
}
GENERIC = {
    "ACL": ["cruciate", "cruzado", "croise", "kruisband", "kreuzband",
            "χιαστ", "кръстн", "capraz bag", "ukrizen"],
    "MCL": ["collateral", "colateral", "kollateral", "collaterale",
            "πλαγι", "колатерал", "yan bag"],
    "Medial Meniscus": ["menisc", "menisk", "μηνισκ", "мениск"],
    "Lateral Meniscus": ["menisc", "menisk", "μηνισκ", "мениск"],
}
SIDE_Q = {
    "ACL": ["anterior", "anterieur", "voorste", "vorder", "προσθι", "предн", "prednj"],
    "MCL": MEDIAL_Q,
    "Medial Meniscus": MEDIAL_Q,
    "Lateral Meniscus": LATERAL_Q,
}
DIRECT = {
    "Effusion": [
        "effusion", "joint fluid", "hemarthrosis", "haemarthrosis", "hydrops",
        "derrame", "epanchement", "gewrichtsvocht", "vocht in het gewricht",
        "erguss", "gelenkerguss", "gelenkserguss",
        "eklem ici sivi", "sivi artisi", "efuzyon", "eklem mesafesinde sivi",
        "eklem ici serbest sivi", "eklem sivisi",
        "ενδαρθρικ", "αρθρικο υγρο", "συλλογη υγρου",
        "ставен излив", "излив", "хидропс", "zglobni izljev", "izljev"],
    "Synovitis": [
        "synovitis", "synovial thickening", "synovial hypertrophy",
        "thickened synovial", "hypertrophy of the synovium",
        "synovial proliferation", "proliferation of the synovium",
        "sinovitis", "synovite", "synovitiden", "synovialitis",
        "verdikking van het synovium", "verdikkingen van het synovium",
        "synoviale verdikking", "synovialisverdickung", "synovialverdickung",
        "sinovit", "hoffitis", "sinovyal kalinlasma", "sinovyal proliferasyon",
        "συνοβιτ", "υμενιτ", "синовит", "sinovij", "sinovije"],
    "Baker's": [
        "baker", "popliteal cyst", "poplitealcyst", "popliteal cysts",
        "quiste popliteo", "quistes popliteos", "quiste de baker",
        "kyste de baker", "kyste poplite", "popliteale cyste", "popliteale cyst",
        "bakercyste", "bakerzyste", "poplitealzyste", "popliteazyste",
        "baker kisti", "popliteal kist",
        "κυστη baker", "κυστη του baker", "киста на бейкър",
        "бейкърова киста", "poplitealna cista"],
    "Contusion": [
        "contusion", "contusiones", "contusie", "kontusyon", "kontuzyon",
        "bone bruise", "bone bruising", "knochenprellung", "prellung",
        "botcontusie", "μωλωπ", "θλαση", "контузи", "kontuzij", "nagnjecen"],
    "Fracture": [
        "fracture", "fractur", "fractura", "fraktur", "fractuur", "breuk",
        "kirik", "kirig", "καταγμα", "καταγματ", "фрактура", "счупван",
        "avulsion", "avulsie", "avulsiyon", "prijelom"],
}
NEG_SRC = [
    r"\bno\b", r"\bnot\b", r"\bnon\b", r"\bwithout\b", r"\babsen\w*",
    r"\bnegative for\b", r"\bunremarkable\b", r"\bintact\w*", r"\bnormal\w*",
    r"\bpreserved\b", r"\bfree of\b", r"\bexcluded\b",
    r"\bno hay\b", r"\bsin\b", r"\bausen\w*", r"\bno se\b", r"\bconservad\w*",
    r"\bintegr\w*", r"\bindemne\b", r"\bpas de\b", r"\baucun\w*", r"\bsans\b",
    r"\bgeen\b", r"\bzonder\b", r"\bonopvallend\w*", r"\bnormaal\b",
    r"\bvrij van\b", r"\bintacte\b",
    r"\bkein\w*", r"\bohne\b", r"\bunauffallig\w*", r"\bregelrecht\w*",
    r"\bnicht\b", r"\bintakt\w*",
    r"\byok\w*", r"\bizlenmemis\w*", r"\bizlenmedi\w*", r"\bsaptanmamis\w*",
    r"\bgozlenmemis\w*", r"\bgorulmemis\w*", r"\bkorunmus\w*",
    r"\bmevcut degil\b", r"\bnormaldir\b", r"\bdogaldir\b",
    r"\bδεν\b", r"\bχωρις\b", r"\bφυσιολογικ\w*", r"\bακεραι\w*",
    r"\bбез\b", r"\bняма\b", r"\bне се\b", r"\bнормал\w*", r"\bзапазен\w*",
    r"\bсъхранен\w*", r"\bsenza\b", r"\bsem\b", r"\bnao\b", r"\bbez\b",
    r"\buredn\w*",
]

NEG_RE = re.compile("|".join(NEG_SRC))
NEG_BACK = 70
NEG_FWD = 45
QUAL_WINDOW = 90
PAIR_WINDOW = 160

TRAUMA_SRC = [
    r"\bfracture\w*", r"\bfractur\w*", r"\bfraktur\w*", r"\bfractuur\w*",
    r"\bbreuk\w*", r"\bkirik\w*", r"\bavuls\w*", r"\bimpaction\w*",
    r"\bcontus\w*", r"\bcontusion\w*", r"\bbruise\w*", r"\bbruising\b",
    r"\bkontuz\w*", r"\bprellung\w*", r"\bkwetsing\w*", r"\btrauma\w*",
    r"\bakut\w*", r"\bacute\w*", r"\bagud\w*", r"\boedem\w*", r"\bedem\w*",
    r"\bödem\w*", r"\bodem\w*", r"\bedema\w*", r"\bedematoz\b",
    r"\bödematöz\b",
    r"\bκαταγμ\w*", r"\bфрактур\w*", r"\bконтуз\w*", r"\bneglect\w*",
    r"\bsubchondral fracture\w*", r"\bosteochondral injur\w*",
]
TRAUMA_RE = re.compile("|".join(TRAUMA_SRC))
CLAUSE_SPLIT = re.compile(r"[.;:\n\r•·]+|\s-\s|\s>\s|\s\*\s")
PF_MASK = re.compile(
    r"(?:medial|lateral)\s+(?:patellar|patella|facet|faset|trochlea|"
    r"trochlear|retinac)\w*"
    r"|(?:medial|lateral)\w*\s+(?:faset|facet|patel|patell|trochl|retinac)\w*"
    r"|(?:patellar|patella|facet|faset|trochlea|trochlear|retinac|rotul|"
    r"patelo)\w*\s+(?:medial|lateral|superior|inferior)"
    r"|(?:patellofemoral|patelofemoral|femoropatellar|femoropatelar|"
    r"retinacul\w*|retinakul\w*|rotulian\w*|patellar|patella)")

STATUS_NONE = 0
STATUS_NEG = 1
STATUS_POS = 2


def fold_text(text):
    t = str(text).lower().translate(_CHAR_MAP)
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c))


def _find_any(clause, terms):
    best = -1
    for term in terms:
        i = clause.find(term)
        if i >= 0 and (best < 0 or i < best):
            best = i
    return best


def _negated(clause, pos, span):
    back = clause[max(0, pos - NEG_BACK):pos]
    fwd = clause[pos + span:pos + span + NEG_FWD]
    return bool(NEG_RE.search(back) or NEG_RE.search(fwd))


def _direct_status(clause, terms):
    seen = False
    for term in terms:
        start = 0
        while True:
            i = clause.find(term, start)
            if i < 0:
                break
            seen = True
            if not _negated(clause, i, len(term)):
                return STATUS_POS
            start = i + 1
    return STATUS_NEG if seen else STATUS_NONE


def _anatomy_index(clause, label):
    i = _find_any(clause, SPECIFIC[label])
    if i >= 0:
        return i
    g = _find_any(clause, GENERIC[label])
    if g >= 0 and _find_any(clause, SIDE_Q[label]) >= 0:
        return g
    return -1


def _paired_status(clause, label, finds):
    seen = False
    ai = _anatomy_index(clause, label)
    if ai >= 0:
        for term in finds:
            fi = clause.find(term)
            while fi >= 0:
                if abs(fi - ai) <= PAIR_WINDOW:
                    seen = True
                    if not _negated(clause, fi, len(term)):
                        return STATUS_POS
                fi = clause.find(term, fi + 1)
    return STATUS_NEG if seen else STATUS_NONE


def _bump(best, cur):
    return max(best, cur)


def _compartment_status(clause, qualifiers, mask_pf):
    if TRAUMA_RE.search(clause):
        return STATUS_NONE
    best = STATUS_NONE
    seen = False
    for m in re.finditer("|".join(map(re.escape, OA_FIND)), clause):
        oi = m.start()
        if _negated(clause, oi, 8):
            continue
        seen = True
        win = clause[max(0, oi - QUAL_WINDOW):oi + QUAL_WINDOW]
        if mask_pf:
            win = PF_MASK.sub(" ", win)
        if _find_any(win, qualifiers) >= 0:
            best = STATUS_POS
    if _find_any(clause, TRICOMP) >= 0:
        seen = True
        if not _negated(clause, max(0, _find_any(clause, TRICOMP)), 8):
            best = STATUS_POS
    return best if best == STATUS_POS else (STATUS_NEG if seen else STATUS_NONE)


def extract_status(report):
    out = {lab: STATUS_NONE for lab in TARGETS}
    if not isinstance(report, str) or not report.strip():
        return out
    text = fold_text(report)
    for raw in CLAUSE_SPLIT.split(text):
        clause = raw.strip()
        if len(clause) < 3:
            continue
        for lab, terms in DIRECT.items():
            if out[lab] != STATUS_POS:
                out[lab] = _bump(out[lab], _direct_status(clause, terms))
        tear = TEAR + SPRAIN
        for lab in SPECIFIC:
            if out[lab] == STATUS_POS:
                continue
            finds = tear if lab in ("ACL", "MCL") else TEAR
            out[lab] = _bump(out[lab], _paired_status(clause, lab, finds))
        for lab, quals in (("Medial OA", MEDIAL_Q), ("Lateral OA", LATERAL_Q),
                           ("PF OA", PF_Q)):
            if out[lab] != STATUS_POS:
                out[lab] = _bump(out[lab], _compartment_status(
                    clause, quals, mask_pf=lab != "PF OA"))
    return out


def status_to_target(status):
    value = 1.0 if status == STATUS_POS else 0.0
    if status == STATUS_POS:
        conf = POSITIVE_CONF
    elif status == STATUS_NEG:
        conf = MENTION_CONF
    else:
        conf = MISSING_CONF
    return value, conf


def extract_labels(report):
    status = extract_status(report)
    return {lab: status_to_target(status[lab]) for lab in TARGETS}


def build_label_table(train_csv):
    train = pd.read_csv(train_csv)
    cols = list(train.columns)
    rows = []
    for vals in train.itertuples(index=False, name=None):
        rec = dict(zip(cols, vals))
        labels = extract_labels(rec.get("Report"))
        gold = not pd.isna(rec.get("ACL"))
        row = {"StudyInstanceUID": rec["StudyInstanceUID"], "is_gold": int(gold)}
        for lab in TARGETS:
            value = float(rec[lab]) if gold else labels[lab][0]
            row[lab] = value
            row[lab + "__conf"] = 1.0 if gold else labels[lab][1]
        rows.append(row)
    return pd.DataFrame(rows, columns=["StudyInstanceUID", "is_gold"] +
                        [c for lab in TARGETS for c in (lab, lab + "__conf")])


def targets_from_table(table):
    values = table[TARGETS].to_numpy(dtype=np.float32)
    conf = table[[lab + "__conf" for lab in TARGETS]].to_numpy(dtype=np.float32)
    gold = table["is_gold"].to_numpy(dtype=np.float32)
    return values, conf, gold


def save_label_table(train_csv, out_path):
    table = build_label_table(train_csv)
    table.to_csv(out_path, index=False)
    return table


def load_label_table(path):
    return pd.read_csv(path)


CONSENSUS_FILE = "llm_labels_v4_blend.csv"


def resolve_consensus(spec):
    if spec != "auto":
        p = Path(spec)
        if not p.is_file():
            raise FileNotFoundError(f"consensus labels not found: {spec}")
        return p
    roots = [Path("/kaggle/input"), Path.cwd()]
    hits = []
    for root in roots:
        if root.is_dir():
            hits.extend(sorted(root.glob("**/" + CONSENSUS_FILE)))
    if not hits:
        raise FileNotFoundError(
            f"{CONSENSUS_FILE} not found under /kaggle/input or cwd")
    return hits[0]


def apply_consensus(table, partner_path):
    """Downweight cells where a second labeller disagrees with our values.

    Values are untouched (validation truth stays comparable); only the
    training confidence of disagreement cells drops to MISSING_CONF, and
    gold rows are never altered. Returns (table, stats).
    """
    partner = pd.read_csv(partner_path, dtype={"StudyInstanceUID": str})
    idx = partner.set_index("StudyInstanceUID")
    out = table.copy()
    gold = out["is_gold"].to_numpy() == 1
    conf_cols = [lab + "__conf" for lab in TARGETS]
    conf = out[conf_cols].to_numpy(dtype=np.float64).copy()
    before = conf.copy()
    stats = {"partner": Path(partner_path).name, "n_gold_skipped": int(gold.sum()),
             "down": MISSING_CONF, "per_label": {}}
    for j, lab in enumerate(TARGETS):
        if lab not in idx.columns:
            raise KeyError(f"{lab} missing from {partner_path}")
        vals = out["StudyInstanceUID"].map(idx[lab]).astype(float).to_numpy()
        present = ~np.isnan(vals)
        hard = (vals > 0.5) != (out[lab].to_numpy(dtype=float) > 0.5)
        mask = present & hard & ~gold
        conf[mask, j] = MISSING_CONF
        stats["per_label"][lab] = int(mask.sum())
    for j, lab in enumerate(TARGETS):
        out[lab + "__conf"] = conf[:, j]
    stats["n_down"] = int(np.count_nonzero(conf != before))
    return out, stats


def label_columns():
    return TARGETS + [lab + "__conf" for lab in TARGETS] + ["is_gold"]


def audit(table, train_csv):
    train = pd.read_csv(train_csv)
    gold = train.dropna(subset=["ACL"])
    truth = gold[TARGETS].to_numpy(dtype=np.float64) > 0.5
    rule_rows = []
    for rep in gold.Report:
        st = extract_status(rep)
        rule_rows.append([float(st[lab] == STATUS_POS) for lab in TARGETS])
    rule = np.asarray(rule_rows, dtype=np.float64).reshape(-1, len(TARGETS)) > 0.5
    if truth.size == 0:
        return {"n_gold": 0, "macro_agreement": float("nan"),
                "per_label_agreement": {lab: float("nan") for lab in TARGETS},
                "n_labels": N_LABELS, "label_file": LABEL_FILE}
    per_label = dict(zip(TARGETS, (truth == rule).mean(axis=0).tolist()))
    return {
        "n_gold": int(len(gold)),
        "macro_agreement": float((truth == rule).mean()),
        "per_label_agreement": per_label,
        "n_labels": N_LABELS,
        "label_file": LABEL_FILE,
    }
