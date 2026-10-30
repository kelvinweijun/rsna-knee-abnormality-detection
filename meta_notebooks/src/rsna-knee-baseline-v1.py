# %% cell 0
from pathlib import Path
from IPython.display import Image, display


def _find_cover(name="RSNA_KNEE_1.png"):
    """Locate the cover image wherever its dataset was mounted.

    A hardcoded mount path guarded by `exists()` is the worst of both worlds: get it
    wrong and the image simply is not there, with nothing said. Kaggle also does not
    always mount a dataset at the same depth. Searching the attached inputs costs one
    directory listing per input and cannot fail quietly. The competition mount is
    skipped by inspection rather than by name, because it holds hundreds of thousands
    of files and none of them is this one.
    """
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        if any((d / s).is_dir() for s in ("train_series", "test_series")):
            continue
        hit = next(d.rglob(name), None)
        if hit is not None:
            return hit
    return None


_cover = _find_cover()
if _cover is not None:
    display(Image(filename=str(_cover)))



# %% [markdown]
# # Twelve findings from one knee MRI
# 
# Each study is a set of MRI series acquired in one session, and the task is to give it
# twelve probabilities: anterior cruciate and medial collateral ligament injury, medial and
# lateral meniscal tear, osteoarthritis in each of the three compartments, joint effusion,
# synovitis, Baker's cyst, bone contusion, and fracture.
# 
# The sections are ordered by how the decisions constrain each other: what the score rewards
# decides how predictions are combined, where the targets come from decides what can be
# trained, and what the scanner recorded decides what the encoder is shown.
# 


# %% [markdown]
# > **On the two label sources.** §2 describes two readers for one job: a rule extractor,
# > defined in full below, and a language model reading the same reports, whose output is a
# > public table attached as a dataset. With the table mounted it supplies the targets,
# > without it the extractor does, and every cell after that is unchanged.
# 


# %% [markdown]
# ## 1. What the score rewards
# 
# The score is the unweighted mean of twelve per-label ROC AUCs:
# 
# $$\text{Score} \;=\; \frac{1}{12}\sum_{i=0}^{11} \mathrm{AUC}_i .$$
# 
# Three consequences follow, and each removes a design choice.
# 
# **Only order matters.** $\mathrm{AUC}_i$ is invariant under any strictly increasing map of
# the scores for label $i$, so calibration and thresholds are worth nothing. It also fixes
# how to combine models: averaging raw probabilities lets whichever model is most confident
# dominate, whereas averaging *ranks* combines the only information the metric reads.
# 
# **Every label costs the same.** Write $M$ for the mean AUC a good model could reach. A
# label left at chance contributes $0.5$ instead of roughly $M$, forfeiting $(M-0.5)/12$ of
# the final score however well the other eleven do — at $M = 0.85$, $0.029$. Rare findings
# deserve *more* attention than common ones, because a rare finding is where a model most
# easily ends up at chance.
# 
# **Prevalence drifts are survivable, thresholds are not.** AUC is in expectation invariant
# to the positive rate, and the competition states prevalence is not guaranteed to match
# across the training, public and final sets. One cutoff survives below — the graded targets
# are binarised at their midpoint so that a held-out AUC can be computed at all — and it
# decides which epoch and configuration are kept, never a submitted score.
# 


# %% [markdown]
# ## 2. Where the targets come from
# 
# Only a small subset of training studies carry the twelve per-condition labels. Every
# training study carries the original radiology report, and the data description invites
# deriving labels from it.
# 
# The decisive fact is in the schemas rather than the prose: `train.csv` has a `Report`
# column and `test.csv` does not. Text is available when fitting and absent when predicting.
# That rules out a fusion model with a text branch — at inference it would have nothing to
# read — and leaves the reports usable only as training targets, as an auxiliary signal
# dropped at inference, or as a weight on how confidently a study could be read. This
# notebook takes the first and the third: a multilingual rule extractor reads each report
# clause by clause, deciding for each finding whether the clause asserts, negates or hedges
# it, and emits a score with a confidence. The confidence becomes a sample weight, so a study
# whose report says nothing about synovitis pulls on that head far less than one that names it.
# 
# **Two readers.** A lexicon matches morphology, so its failure mode is silence rather than
# error, and silence is measurable without ground truth: for each (report, finding) pair, ask
# only whether anything matched at all. That rate needs no annotations, so it runs on every
# study, and tagged by language it says *where* the vocabulary is thin. Here the misses are
# concentrated — one language is covered far better than the other eight, and the gap falls on
# findings a knee report almost always comments on. Enumerating morphology for nine languages
# is the wrong instrument for that; reading the sentence is the right one, and a language model
# reads it, asked for the same twelve findings in the same graded form. The annotated studies
# decide between the readers, paired on the same studies, and the difference is large and
# one-sided in the direction the coverage rate predicts. So the pipeline prefers a mounted
# table of model-read labels when present and runs the lexicon when not; both emit the same
# columns, and a partial table falls back per study rather than per run.
# 
# Two details matter more than either reader's internals.
# 
# **Reports are graded, annotations are thresholded.** The reporting radiologist and the
# annotator do not share a threshold. A report saying *small joint effusion* may sit against
# a negative annotation, because the annotator marked only effusions they judged significant.
# A rule of the form *term present $\Rightarrow$ positive* is wrong by construction; grading
# the mention — trace, unqualified, marked — is right and costs nothing, because §1
# established that only order is read.
# 
# **Derived labels are not independent across studies.** A report shared verbatim by several
# studies yields one target vector for all of them, which must be respected when splitting;
# §7 does.
# 


# %% [markdown]
# ### Reading a report in nine languages
# 
# **No language is identified.** Every cue lexicon carries all nine languages at once and
# each clause is tested against the union. Routing first means committing to a guess before
# any evidence is read, and the cheap guess — substring tests, `'the '` for English, `'la '`
# for French — fails badly, because `la` is as common in Spanish as in French and whichever
# test runs first swallows both. Pooling costs little, since Greek and Cyrillic cues cannot
# collide with Latin-script ones and the Latin-script vocabularies of interest are close
# enough that a shared cue is usually right. The price is paid in coverage instead.
# 
# **Normalise, then segment, then scope.** Case, diacritics and separators are folded first,
# which also repairs a codepoint problem: many Greek reports spell mu with MICRO SIGN U+00B5
# rather than U+03BC, and NFKD maps one onto the other. Text is then split into clauses, with
# a heading line attached to the value beneath it, because a report reading `Fractures :`
# then `Aucune.` states one thing across two lines and any method that splits them reads a
# negation as a positive.
# 
# **Assertion, negation, hedge.** Negation is not an edge case: for several findings most
# mentions are negative, since a report lists what was checked and found intact. Explicit
# normality counts as negation — *ligamentos cruzados y colaterales dentro de límites
# normales* is evidence of absence, not absence of evidence — except where a tear or a high
# grade is named in the same breath.
# 
# **Stems behind phrases.** Four targets need an anatomy word and a pathology word together.
# A lexicon of complete phrases carries most matches but cannot survive morphology: Turkish
# suffixes possessives onto the noun, Croatian and Greek decline it. Where the phrase fails,
# a second pass matches a stem and requires a side qualifier within a *character* window,
# which handles inflection without enumerating it, and handles word order — which puts the
# side adjective before the noun in English and after it in Greek.
# 
# **Why coverage decides and agreement does not.** A rule that never fires raises no error;
# in a binary extractor it emits a negative, indistinguishable from a confident one, and a
# lexicon complete in English and thin in Greek looks like a corpus where Greek patients have
# fewer findings. The obvious check — agreement with the per-condition annotations — measures
# the right thing on far too few studies. The Hanley–McNeil standard error of an AUC $A$ with
# $n_p$ positives and $n_n$ negatives,
# 
# $$
# \mathrm{SE}(A)=\sqrt{\frac{A(1-A)+(n_p-1)(Q_1-A^{2})+(n_n-1)(Q_2-A^{2})}{n_p\,n_n}},
# \qquad
# Q_1=\frac{A}{2-A},\quad Q_2=\frac{2A^{2}}{1+A},
# $$
# 
# at $A\approx0.8$ and a handful of positives among a few dozen studies lands near $0.09$ — a
# 95% interval of roughly $\pm0.17$ — far wider than the differences between two lexicons it would be asked to
# separate. The silence rate has no such limit, so it is what points at the missing
# vocabulary and what changes to the lexicon are judged on.
# 


# %% cell 6
from __future__ import annotations

import re
import unicodedata

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


# Turkish dotted/dotless i must be folded before casefolding, otherwise "İZLENMEZ"
# and "izlenmez" diverge. ß and the Croatian/Serbian d-with-stroke likewise.
_PRE = str.maketrans({
    "ı": "i", "İ": "i", "I": "i", "ß": "ss", "đ": "d", "Đ": "d",
    "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae",
})


def normalize(text: str) -> str:
    """Fold case, diacritics and separators; keep Greek and Cyrillic letters.

    NFKD decomposition strips Latin accents and Greek tonos alike (ά -> α), which is what
    we want: reports are inconsistent about accents. It also maps the MICRO SIGN U+00B5
    to a real mu, which matters because most Greek reports here use the wrong codepoint.
    """
    if not isinstance(text, str):
        return ""
    text = text.translate(_PRE).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("­", "")                    # soft hyphen
    text = re.sub(r"[_\-/\\]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


_SENT_SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+")


def clauses(text: str):
    """Split into clauses, then attach `header:` lines to the value that follows.

    A report line reading `Fractures :` followed by `Aucune.` is one statement. Splitting
    on punctuation alone separates the anatomy from its negation and flips the label.
    """
    norm = normalize(text)
    raw = [c.strip() for c in _SENT_SPLIT.split(norm) if c and c.strip()]

    merged = []
    for i, c in enumerate(raw):
        # A fragment ending in a colon is a heading for the next fragment. Structured
        # English reports write long ones - "lateral compartment (meniscus, collateral
        # ligament complex, cartilage):" is eight words - so the cap is generous.
        #
        # A merged heading must NOT also stand alone. On its own it carries the anatomy
        # word with no negation in scope, so `Fractures :` / `Aucune.` asserted a fracture
        # off the heading while the joined clause correctly read the denial. The joined
        # clause is a superset of the heading, so nothing is lost by dropping it; a
        # heading with no value beneath it is not merged and still stands.
        if c.endswith(":") and len(c.split()) <= 14 and i + 1 < len(raw):
            merged.append(c + " " + raw[i + 1])
        else:
            merged.append(c)
    # Comma-separated enumerations inside a long clause hide separate assertions.
    out = []
    for c in merged:
        out.append(c)
        if len(c.split()) > 25:
            out.extend(p.strip() for p in c.split(",") if len(p.split()) > 2)
    return out


def _rx(*alts: str) -> re.Pattern:
    return re.compile("|".join(alts))



# %% cell 7
NEGATION = _rx(
    # en
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b", r"\babsence\b",
    r"\bno evidence\b", r"\bunremarkable\b", r"\bfree of\b", r"\bnone\b", r"\bnil\b",
    # es
    r"\bsin\b", r"\bno hay\b", r"\bausencia\b", r"\bausentes?\b",
    # fr
    r"\bpas de\b", r"\bsans\b", r"\baucune?\b", r"\babsence\b",
    # nl
    r"\bgeen\b", r"\bzonder\b", r"\bniet\b",
    # de
    r"\bkeine?\b", r"\bohne\b", r"\bnicht\b",
    # tr
    r"\byok\b", r"\byoktur\b", r"izlenmemekte", r"saptanmadi", r"\bdegil\b",
    r"gozlenmemekte", r"mevcut degil", r"eslik etmiyor", r"\bizlenmedi\b",
    # hr / sr / bs
    r"\bnema\b", r"\bbez\b", r"\bnisu\b", r"\bnije\b",
    # el (accents already stripped)
    r"\bδεν\b", r"\bχωρις\b", r"ουδεν",
    # bg / ru
    r"\bбез\b", r"\bне\b", r"липсва", r"\bняма\b",
)

NORMALITY = _rx(
    r"\bnormal", r"\bintact\b", r"\bpreserved\b", r"\bwithin normal limits\b",
    r"limites normales", r"\bconservad", r"\bintegr", r"\bnormales\b",
    r"\bdoga(l|ll)\b", r"korunmus", r"\bnormaldir\b", r"olagan",
    r"\buredn", r"\bocuvan", r"\bodrzan", r"\bintakt",
    r"φυσιολογικ", r"ακεραι",
    r"unauffallig", r"regelrecht", r"\bintakt\b",
    r"нормал", r"запазен", r"съхранен", r"\bбез особености\b",
    r"\bgaaf\b", r"\bnormaal\b",
)

UNCERTAIN = _rx(
    r"\bpossible\b", r"\bprobable\b", r"\bsuspicious\b", r"\bsuspected\b",
    r"cannot (be )?exclude", r"\bmay\b", r"\bquestionable\b", r"\bequivocal\b",
    r"\bposible\b", r"sin criterios categoricos", r"\bdudos",
    r"\bmuhtemel\b", r"\bolasi\b", r"\bsupheli\b", r"\bizlenim",
    r"\bmoguce\b", r"\bvjerojatno\b", r"\bsumnja\b",
    r"πιθαν", r"υποπτ",
    r"\bmoglich", r"\bverdachtig", r"\bfraglich", r"\bV\.a\.\b",
    r"\bвъзможно\b", r"\bвероятно\b", r"суспект",
    r"\bmogelijk\b", r"\bverdacht\b",
)

# Pathology vocabulary shared by the paired rules.
TEAR = _rx(
    r"\btear", r"\btorn\b", r"\brupture", r"\bdisruption\b", r"discontinuit",
    r"\bavuls",
    r"\brotura\b", r"\broturas\b", r"\bruptura", r"\bdesgarro", r"\broto\b",
    r"\bdechirure", r"\bdechire",
    r"\bscheur", r"\bruptuur", r"gescheurd",
    r"riss(bildung|e|es)?\b", r"einriss", r"\bruptur", r"zerreiss", r"\blasion",
    r"\byirtik", r"\byirtig", r"\bkopma\b", r"butunluk kaybi", r"\brupturu\b",
    r"\bpuknuce", r"\bruptur", r"\bprekid\b", r"\bpukotin",
    r"ρηξη", r"ρηξις", r"ρηγμα",
    r"руптура", r"разкъсв", r"разрив", r"скъсв",
)

DEGEN = _rx(
    r"degenerat", r"\bmucoid\b", r"\bmyxoid\b", r"\bfray", r"\bfissur",
    r"dejeneratif", r"\bmukoid\b", r"degenerativn", r"εκφυλ", r"дегенерат",
    r"\bμυξοειδ", r"\bμυξωδ",
    r"\bmuco ?ide\b", r"aufgefasert",
)

INJURY = _rx(
    r"\binjur", r"\bsprain", r"\blesion", r"\blasion", r"\bedema\b", r"\boedema\b",
    r"\bodem\b", r"\bedem\b", r"\bοιδημα", r"\bодем", r"\bедем", r"\bstrain\b",
    r"\bhigh signal\b", r"\bsignal alteration\b", r"\bhiperintens", r"\bhyperintens",
    r"aumento de senal", r"alteracion de senal", r"cambio de senal",
    r"\bsignalanhebung", r"\bsignalalteration", r"verhoogd signaal", r"sinyal artis",
    r"αυξημενο σημα", r"повишен сигнал",
    r"\bthicken", r"\bzadebljanje\b", r"\bverdikking\b", r"\bdistenzij",
    r"\blaksite\b", r"\blaxity\b", r"\bpartial\b", r"\bparcijaln", r"\bparcial",
    r"\bpartiel", r"\bpartiell",
)



# %% cell 8
ANAT = {
    "ACL": _rx(
        r"anterior cruciate", r"\bacl\b",
        r"cruzado anterior", r"\blca\b",
        r"croise anterieur",
        r"voorste kruisband", r"\bvkb\b",
        r"vorderes kreuzband", r"vorderen kreuzband", r"vordere kreuzband",
        r"on capraz", r"\bocb\b",
        r"prednji krizni", r"prednjeg krizn",
        r"προσθι[οα][^ ]* χιαστ", r"προσθιου χιαστου", r"χιαστο[^ ]* συνδεσμ",
        # "χιαστοι και πλαγιοι συνδεσμοι" separates the adjective from its noun, so the
        # adjective stem has to stand alone. Greek marks cruciate with it unambiguously.
        r"\bχιαστ\w*",
        r"предна кръстна", r"предната кръстна",
        # Plural, unqualified: reports routinely clear both cruciates in one clause
        # ("Ligamentos cruzados y colaterales dentro de limites normales"), so the
        # plural form has to match without a side qualifier or the whole clause is lost.
        r"cruciate ligaments", r"ligamentos cruzados", r"ligaments croises",
        r"kruisbanden", r"kreuzbander", r"capraz baglar", r"krizn[a-z]* ligament[a-z]*",
        r"χιαστοι συνδεσμ", r"χιαστων συνδεσμ", r"кръстните връзки", r"кръстни връзки",
    ),
    "MCL": _rx(
        r"medial collateral", r"\bmcl\b", r"tibial collateral",
        r"colateral medial", r"colateral interno", r"\blcm\b",
        r"collateral medial", r"collateral interne",
        r"mediale collaterale", r"binnenband", r"\b(mediale|laterale) banden\b",
        r"\bcollaterale banden\b",
        r"innenband", r"mediales? kollateral",
        r"\bic yan bag", r"medial kollateral", r"\biyb\b",
        r"medijalni kolateraln", r"medijalnog kolateraln",
        r"εσω πλαγι", r"εσωτερικο πλαγι", r"\bπλαγι\w* συνδεσμ", r"\bπλαγιοι\b",
        r"медиален колатерал", r"вътрешна странична", r"\bколатерал\w*",
        # Same plural pattern as the cruciates.
        # "Ligamentos cruzados y colaterales" separates the noun from its adjective, so
        # the adjective has to stand alone as a cue.
        r"\bcolaterales\b", r"\bcollateraux\b", r"\bcollateralen\b", r"\bkolateralni\b",
        r"collateral ligaments", r"ligamentos colaterales", r"ligaments collateraux",
        r"collaterale banden", r"kollateralbander", r"seitenbander", r"yan baglar",
        r"kolateraln[a-z]* ligament[a-z]*", r"πλαγιοι συνδεσμ", r"πλαγιων συνδεσμ",
        r"колатерални връзки", r"страничните връзки",
    ),
    "Medial Meniscus": _rx(
        r"medial meniscus", r"\bmm\b(?= tear)", r"medial menisc",
        r"menisco medial", r"menisco interno",
        r"menisque medial", r"menisque interne",
        r"mediale meniscus", r"binnenmeniscus",
        r"innenmeniskus", r"medialen? meniskus", r"innenmeniskushinterhorn",
        r"medyal menisk", r"\bic menisk",
        r"medijalni meniskus", r"medijalnog meniskusa", r"medijalnom meniskusu",
        r"εσω μηνισκ", r"μηνισκ[^ ]* του εσω", r"εσω διαμερισμα[^.]{0,40}μηνισκ",
        r"медиалния менискус", r"медиален менискус", r"вътрешния менискус",
    ),
    "Lateral Meniscus": _rx(
        r"lateral meniscus", r"lateral menisc",
        r"menisco lateral", r"menisco externo",
        r"menisque lateral", r"menisque externe",
        r"laterale meniscus", r"buitenmeniscus",
        r"aussenmeniskus", r"lateralen? meniskus",
        r"lateral menisk", r"\bdis menisk",
        r"lateralni meniskus", r"lateralnog meniskusa", r"lateralnom meniskusu",
        r"εξω μηνισκ", r"μηνισκ[^ ]* του εξω", r"εξω διαμερισμα[^.]{0,40}μηνισκ",
        r"латералния менискус", r"латерален менискус", r"външния менискус",
    ),
}

# Osteoarthritis is rarely written as "osteoarthritis". It is written as cartilage loss,
# chondropathy grade, joint space narrowing, or osteophytes - scoped to a compartment.
OA_EVIDENCE = _rx(
    r"osteoarthrit", r"\barthros", r"\bgonarthros", r"\bosteoarthros",
    r"chondropath", r"chondromalac", r"condropat", r"condromalac",
    r"cartilage loss", r"cartilage thinning", r"chondral (loss|defect|ulcer|thinning)",
    r"osteophyt", r"osteofit", r"osteofyt", r"osteofito", r"osteophyten",
    r"joint space narrowing", r"pinzamiento articular",
    r"kikirdak kayb", r"kikirdak incelme", r"kondropati", r"kondral",
    r"kraakbeen(lijden|verlies)", r"gonartrose", r"artrose",
    r"knorpel(verlust|schaden|defekt)", r"arthrose", r"gonarthrose",
    r"hrskavic", r"hondromalac", r"artroz", r"osteoartrit",
    r"χονδρ[^ ]*παθ", r"αρθριτ", r"αρθρωσ", r"οστεοφυτ",
    r"αρθρικου χονδρου", r"εξαλειψη του αρθρικου χονδρου",
    r"артроз", r"хондропат", r"остеофит", r"хрущял[^.]{0,30}(изтън|увред|дефект)",
    r"ulcera[s]? condral", r"cartilago[^.]{0,25}(perdida|adelgaz)",
    r"icrs grade", r"outerbridge",
)

COMPARTMENT = {
    "Medial OA": _rx(
        r"medial (femorotibial|tibiofemoral|compartment)",
        r"compartimento femorotibial medial", r"femorotibial interno",
        r"mediaal femorotibiaal", r"mediale femorotibial",
        r"medial femorotibial", r"medialen kompartiment", r"innere[sn]? kompartiment",
        r"medyal femorotibial", r"ic kompartman", r"medyal kompartman",
        r"medijaln[^ ]* (femorotibi|odjelj|kompartm)",
        r"εσω διαμερισμα", r"εσω κνημιαι", r"εσω μηριαι",
        r"медиалн[^ ]* (компартм|отдел|тибиал|феморотиб)",
        r"medial (femoral|tibial) (condyle|plateau)", r"condilo femoral medial",
        r"medialen? (femurkondyl|tibiaplateau)", r"mediale femorale condyl",
    ),
    "Lateral OA": _rx(
        r"lateral (femorotibial|tibiofemoral|compartment)",
        r"compartimento femorotibial lateral", r"femorotibial externo",
        r"lateraal femorotibiaal", r"laterale femorotibial",
        r"lateral femorotibial", r"lateralen kompartiment", r"aussere[sn]? kompartiment",
        r"lateral femorotibial", r"dis kompartman", r"lateral kompartman",
        r"lateraln[^ ]* (femorotibi|odjelj|kompartm)",
        r"εξω διαμερισμα", r"εξω κνημιαι", r"εξω μηριαι",
        r"латералн[^ ]* (компартм|отдел|тибиал|феморотиб)",
        r"lateral (femoral|tibial) (condyle|plateau)", r"condilo femoral lateral",
        r"lateralen? (femurkondyl|tibiaplateau)", r"laterale femorale condyl",
    ),
    "PF OA": _rx(
        r"patellofemoral", r"femoropatellar", r"femoropatelar", r"patelofemoral",
        r"retropatellar", r"retrorotulian", r"\btrochlea", r"\btroclea", r"\btroklea",
        r"\bpatella\b", r"\bpatellar\b", r"\brotulian", r"\brotula\b", r"\bpatele\b",
        r"\bpatellae?\b", r"patellofemoraal", r"femoropatellair",
        r"επιγονατιδ", r"μηροεπιγονατιδ", r"τροχιλ",
        r"пател", r"феморопател", r"тролх",
        r"anterior compartment", r"compartimento anterior", r"prednj[^ ]* odjeljk",
    ),
}

# Self-declaring findings: the term itself is the finding.
DIRECT = {
    "Effusion": _rx(
        r"\beffusion", r"joint fluid", r"intra ?articular fluid", r"\bhydrops\b",
        r"derrame articular", r"\bderrame\b", r"liquido articular",
        r"epanchement",
        r"gewrichtsvocht", r"\bvocht\b", r"\bhydrops\b", r"gewrichtseffusie",
        r"gelenkerguss", r"\berguss\b", r"gelenksergu",
        # "diz eklemi ici sivi miktari ... artmis" and "eklem icerisinde yaygin sivi
        # artisi" both occur; the noun takes a possessive suffix, so `eklem ` alone
        # misses. Match the stem plus any suffix.
        r"eklem\w* ic\w* sivi", r"efuzyon", r"eklem sivisi",
        r"sivi (miktari|artisi|birikimi)", r"sivi artis", r"\bsivi\b[^.]{0,25}artmis",
        r"\bizljev", r"\bizliv", r"zglobn[^ ]* tekucin", r"\bhidrops\b",
        r"αρθρικ[^ ]* υγρ", r"υγρου ενδαρθρικα", r"ενδαρθρικ[^ ]* υγρ", r"ποσοτητα υγρου",
        r"ενδαρθρικ", r"αρθρικη συλλογη", r"υγρο στην αρθρωση", r"υγρου στην αρθρωση",
        r"ставен излив", r"излив", r"ставна течност", r"синовиална течност",
    ),
    "Synovitis": _rx(
        r"synovit", r"sinovit", r"synovial (thickening|proliferation|hypertroph)",
        r"synovitis", r"synoviale? (verdikking|proliferatie)",
        r"synovialitis", r"synovialis(verdickung|proliferation)",
        r"sinovijalitis", r"sinovitis", r"zadebljanje sinovij",
        r"υμενιτιδα", r"συνοβιτιδα", r"υμενικ[^ ]* υπερτροφ", r"αρθρικου υμεν",
        r"синовит", r"синовиал[^ ]* (задебел|пролифер)",
        r"verdikkingen van (het )?synovium", r"pannus",
    ),
    "Baker's": _rx(
        r"baker", r"popliteal cyst", r"quiste popliteo", r"quistes popliteos",
        r"kyste poplite", r"popliteale? cyst", r"poplitealzyste", r"bakerzyste",
        r"popliteal kist", r"\bbakerova\b", r"poplitealn[^ ]* cist",
        r"κυστη baker", r"πολυχωρη συνοβιακη κυστη", r"κυστη του baker",
        r"киста на бейкър", r"бейкърова киста", r"поплитеална киста",
        r"gastrocnemio ?semimembranos", r"gastrocnemius semimembranosus burs",
    ),
    "Contusion": _rx(
        r"\bcontusion", r"bone bruise", r"bone marrow (o?edema|contusion)",
        r"\bkontuz", r"medular bone o?edema", r"marrow o?edema",
        r"contusion osea", r"edema oseo", r"edema de medula osea",
        r"oedeme osseux", r"contusion osseuse",
        r"botcontusie", r"botoedeem", r"beenmergoedeem", r"botmergoedeem",
        r"knochenmarkodem", r"knochenodem", r"kontusion", r"bone bruise",
        r"kemik kontuzyonu", r"kemik iligi odemi", r"kemik odemi",
        r"kostani edem", r"edem kosti", r"kontuzij",
        r"οστεομυελικ[^ ]* οιδημα", r"οστικο οιδημα", r"μυελικο οιδημα",
        r"костномозъчен едем", r"костен едем", r"контузионен",
    ),
    "Fracture": _rx(
        r"\bfractur", r"\bfract\b",
        r"\bfractura", r"\bfracturas\b",
        r"\bfractuur", r"\bbreuk\b",
        r"\bfraktur", r"\bbruch\b",
        r"\bkirik\b", r"\bkirigi\b", r"\bkirik\b",
        r"\bfraktur", r"\bprijelom", r"impresijsk[^ ]* fraktur",
        r"καταγμα", r"καταγματ",
        r"фрактур", r"счупван", r"фисур",
        r"insufficiency fracture", r"stress fracture", r"avulsion fracture",
        r"subchondral fracture", r"subkondral kiri",
    ),
}

# Terms that look like a finding but are not the finding being scored.
DECOY = {
    # `no fracture` is deliberately absent: a decoy skips the clause, so listing it here
    # turned the commonest English denial into silence, and the study then pulled on the
    # fracture head with the weight of a report that never mentioned fractures at all.
    # `microfractur` is a surgical procedure and `fracture risk` a prediction; both stay.
    "Fracture": _rx(r"microfractur", r"\bfracture (risk|prophyla)"),
    "Baker's": _rx(r"meniscal cyst", r"quiste meniscal", r"ganglion"),
}

PAIRED = {"ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"}
OA_TARGETS = {"Medial OA", "Lateral OA", "PF OA"}



# %% cell 9
STEM_MENISCUS = _rx(r"menisc\w*", r"menisk\w*", r"μηνισκ\w*", r"мениск\w*")
STEM_CRUCIATE = _rx(r"cruciate", r"cruzado", r"croise", r"kruisband", r"kreuzband",
                    r"capraz bag\w*", r"krizn\w*", r"χιαστ\w*", r"кръстн\w*",
                    r"\bacl\b", r"\bpcl\b", r"\blca\b", r"\blcp\b", r"\bvkb\b",
                    r"\bhkb\b", r"\bocb\b", r"\bacb\b")
STEM_COLLATERAL = _rx(r"collateral\w*", r"colateral\w*", r"kollateral\w*",
                      r"collaterale\w*", r"kolateraln\w*", r"yan bag\w*",
                      r"πλαγι\w*", r"колатерал\w*", r"странич\w*",
                      r"innenband\w*", r"aussenband\w*", r"binnenband\w*",
                      r"\bmcl\b", r"\blcl\b", r"\blcm\b", r"\biyb\b")

SIDE_MEDIAL = _rx(r"\bmedial\w*", r"\bmedyal\w*", r"\bmedijaln\w*", r"\bmediaal\w*",
                  r"\bmediale\w*", r"\bintern[oa]\w*", r"\binterne\w*", r"\binnen\w*",
                  r"\bic\b", r"\bunutarnj\w*", r"\bεσω\w*", r"\bεσωτερικ\w*",
                  r"\bмедиал\w*", r"\bвътреш\w*", r"\btibial collateral\b",
                  r"\bbinnen\w*", r"\bmediaal\b")
SIDE_LATERAL = _rx(r"\blateral\w*", r"\bextern[oa]\w*", r"\bexterne\w*", r"\bdis\b",
                   r"\blateraln\w*", r"\baussen\w*", r"\bbuiten\w*", r"\bεξω\w*",
                   r"\bεξωτερικ\w*", r"\bлатерал\w*", r"\bвъншн\w*",
                   r"\bfibular collateral\b", r"\bvanjsk\w*")
SIDE_ANTERIOR = _rx(r"\banterior\w*", r"\bant\b", r"\bon\b", r"\bprednj\w*",
                    r"\bvorder\w*", r"\bvoorste\b", r"\bπροσθι\w*", r"\bпредн\w*",
                    r"\banteriyor\w*", r"\bavant\b", r"\bant[eé]rieur\w*")

# The contrary of SIDE_ANTERIOR, needed only to stop a side-blind cruciate cue firing on
# the posterior ligament. It is never used to assert a target - there is no PCL target -
# so it is deliberately narrow: `posterior horn` is one of the commonest phrases in a
# knee report and must not be read as a cruciate qualifier, which is why the guard below
# tests proximity to the cruciate stem rather than presence in the clause.
SIDE_POSTERIOR = _rx(r"\bposterior\w*", r"\bpost[eé]rieur\w*", r"\bposteriore\w*",
                     r"\bhinter\w*", r"\bachterste\b", r"\barka\b", r"\bstraznj\w*",
                     r"\bzadnj\w*", r"\bοπισθι\w*", r"\bзадн\w*", r"\bpostero\w*")

# Fracture is the target whose stem varies most across the corpus.
STEM_FRACTURE = _rx(r"fractur\w*", r"fraktur\w*", r"fractuur\w*", r"\bfract\b",
                    r"kiri[kgğ]\w*", r"prijelom\w*", r"lom kosti", r"\bbreuk\w*",
                    r"\bbruch\w*", r"καταγμα\w*", r"καταγματ\w*", r"фрактур\w*",
                    # NOT a bare `fissur\w*`: "fisuras condrales" and "full thickness
                    # fissures in the articular cartilage" describe cartilage, not bone.
                    # The stem has to be anchored to a bone word to mean a fracture.
                    r"счупван\w*", r"fisur\w* (osea|oseas|kost)", r"fissur\w* kost")

STEM_OA_COMPARTMENT = _rx(r"compartment\w*", r"compartimento\w*", r"compartiment\w*",
                          r"kompartman\w*", r"kompartiment\w*", r"odjelj\w*",
                          r"διαμερισμα\w*", r"компартм\w*", r"\bотдел\w*",
                          r"femorotibial\w*", r"femorotibiaal\w*", r"tibiofemoral\w*",
                          r"femoro tibial\w*", r"κνημιαι\w*", r"μηριαι\w*",
                          r"femoral condyl\w*", r"tibial plateau\w*",
                          r"condilo femoral", r"platillo tibial", r"tibiaplateau\w*",
                          r"femurkondyl\w*", r"femoralne? kondil\w*",
                          r"tibijaln\w* plato", r"femoral kondil\w*",
                          r"tibia plato", r"tibyal plato")


def _distance(clause: str, stem_rx: re.Pattern, qual_rx: re.Pattern, window: int = 55):
    """Characters from the nearest stem to the nearest qualifier, or None if none is near.

    Character windows rather than token windows, because word order differs: English
    puts the side before the noun, Greek and Bulgarian often after, and Turkish
    attaches it as a separate preceding adjective.

    A distance rather than a yes. Presence is enough to decide that a qualifier applies
    to a structure, but not enough to decide which of two qualifiers applies: a knee
    report says "anterior horn" and "posterior horn" constantly, so any window wide
    enough to catch a real side word also catches an unrelated one, and two rules that
    both answer "yes" cannot be told apart. Comparing how far away they are can.
    """
    best = None
    for m in stem_rx.finditer(clause):
        lo = max(0, m.start() - window)
        hi = min(len(clause), m.end() + window)
        for q in qual_rx.finditer(clause[lo:hi]):
            qs, qe = lo + q.start(), lo + q.end()
            d = 0 if qs < m.end() and qe > m.start() else \
                min(abs(m.start() - qe), abs(qs - m.end()))
            best = d if best is None else min(best, d)
    return best


def _near(clause: str, stem_rx: re.Pattern, qual_rx: re.Pattern, window: int = 55):
    """True if a stem match has a qualifier within `window` characters either side."""
    return _distance(clause, stem_rx, qual_rx, window) is not None
    return False


# concept -> (stem, side) pairs used in addition to the phrase lexicons above
STEM_RULES = {
    "ACL": (STEM_CRUCIATE, SIDE_ANTERIOR),
    "MCL": (STEM_COLLATERAL, SIDE_MEDIAL),
    "Medial Meniscus": (STEM_MENISCUS, SIDE_MEDIAL),
    "Lateral Meniscus": (STEM_MENISCUS, SIDE_LATERAL),
    "Medial OA": (STEM_OA_COMPARTMENT, SIDE_MEDIAL),
    "Lateral OA": (STEM_OA_COMPARTMENT, SIDE_LATERAL),
}



# %% cell 10
SEV_LOW = _rx(
    r"\bsmall\b", r"\bminimal\b", r"\btrace\b", r"\bmild\b", r"\bslight\b",
    r"\btiny\b", r"\bscant\b", r"\bmimimal\b", r"\bdiscrete\b", r"\bfocal\b",
    r"\bleve\b", r"\bminim", r"\bpeque", r"\bligero\b", r"\bescaso\b", r"\bdiscreto\b",
    r"\bhafif\b", r"\bminimal\b", r"\baz miktarda\b", r"\bsilik\b",
    r"\bmanja\b", r"\bmanji\b", r"\bblago\b", r"\bdiskretn", r"\bmalo\b",
    r"\bgering", r"\bdiskret", r"\bkleine?r?\b", r"\bwenig\b", r"\bzarte?\b",
    r"\bbeperkte?\b", r"\bgeringe\b", r"\bweinig\b", r"\blichte?\b",
    r"\bηπι", r"\bμικρ", r"\bελαχιστ",
    r"\bминимал", r"\bлек", r"\bмалк", r"\bнеголям",
)

SEV_HIGH = _rx(
    r"\blarge\b", r"\bmarked\b", r"\bmassive\b", r"\bsevere\b", r"\bextensive\b",
    r"\bmoderate\b", r"\bgross\b", r"\bsignificant\b", r"\babundant\b", r"\btense\b",
    r"\bmoderad", r"\bimportante\b", r"\bsevera?\b", r"\bmarcad", r"\bcuantios",
    r"\bbelirgin\b", r"\byaygin\b", r"\bileri\b", r"\bciddi\b", r"\bbol\b",
    r"\bopsezan\b", r"\bveliki\b", r"\bizrazit", r"\bznacajn", r"\bumjeren",
    r"\bausgepragt", r"\bdeutlich", r"\bmassiv", r"\bmassig", r"\bgross",
    r"\buitgebreid", r"\bgevorderd", r"\bveel\b", r"\bmatige?\b",
    r"\bμετρι", r"\bμεγαλ", r"\bεκτεταμεν", r"\bευμεγεθ", r"\bσοβαρ",
    r"\bголям", r"\bизразен", r"\bзначим", r"\bумерен", r"\bобилен",
)

# OA is often asserted for the whole joint rather than per compartment
# ("tricompartmental osteoarthritis", "gonarthrose", "incipient OA of all three
# compartments"). Those statements are evidence for all three OA targets.
GLOBAL_OA = _rx(
    r"tri ?compartment", r"all three compartment", r"global(ised)? (oa|osteoarthrit)",
    r"\bgonarthros", r"\bgonartros", r"\bgonarthrose", r"\bgonartrose",
    r"osteoarthritis of the knee", r"artrosis (de |)(la )?rodilla", r"knee osteoarthrit",
    r"\bdiz osteoartrit", r"\bgonartroz", r"artroza koljena",
    r"οστεοαρθριτιδα", r"αρθριτιδα του γονατος",
    r"артроза на колянната", r"гонартроз",
    r"degenerative joint disease", r"\bdjd\b",
)

# A bare "bone marrow oedema" is not a contusion when it sits under a cartilage
# defect: subchondral oedema beneath a worn compartment is reactive degenerative signal,
# and reading it as a bruise turns every osteoarthritic knee into a trauma case.
DEGENERATIVE_MARROW = _rx(
    r"subchondral", r"subcondral", r"subkondral", r"supkondraln", r"subchondraln",
    r"υποχονδρι", r"субхондрал", r"subchondrale?",
    r"\bcyst", r"\bquist", r"\bzyste\b", r"\bcistic", r"reactive", r"reactivo",
)

TRAUMA = _rx(
    r"\bbruise\b", r"\bcontusion", r"\bkontuz", r"\bcontusion osea\b",
    r"\btrauma", r"\bimpaction\b", r"\bpivot shift\b", r"\bkissing\b",
    r"\bacute\b", r"\bagudo\b", r"\bakut", r"\bpivot kaymasi\b",
    r"\bcontusion osseuse\b", r"\bbone bruise\b", r"\bbotcontusie\b",
    r"\bконтузион", r"\bμωλωπ", r"\bkontuzij",
)



# %% cell 11
def _polarity(clause: str, anchor_end: int) -> str:
    """Classify one clause as positive, negative or uncertain for a matched term.

    Scope is the whole clause. Clause segmentation already keeps statements short, and
    a window in characters mis-scopes badly across languages with different word orders -
    Turkish puts its negator at the end of the sentence, English at the front.
    """
    if UNCERTAIN.search(clause):
        return "uncertain"
    if NEGATION.search(clause):
        return "negative"
    if NORMALITY.search(clause):
        # "meniscus normal" negates; "normal ... but tear" does not.
        if TEAR.search(clause) or re.search(r"\bgrade [34]\b", clause):
            return "positive"
        return "negative"
    return "positive"


class _Matcher:
    """Phrase lexicon first, stem+side proximity as the fallback.

    Exposes `.search` so it drops into the same slot as a compiled pattern.
    """

    def __init__(self, phrase_rx, stem=None, side=None, window=55, contrary=None):
        self.phrase_rx = phrase_rx
        self.stem = stem
        self.side = side
        self.window = window
        self.contrary = contrary

    def search(self, clause):
        m = self.phrase_rx.search(clause)
        if m is not None and not self._wrong_side(clause):
            return m
        if self.stem is not None and _near(clause, self.stem, self.side, self.window):
            return self.stem.search(clause)
        return None

    def _wrong_side(self, clause):
        """True when the clause names the other member of this structure's pair.

        Some cues in the lexicon are side-blind by design: Greek separates the adjective
        from its noun ("cruciate and collateral ligaments"), so the bare adjective stem
        has to stand alone or the clause is lost. That stem then also matches the
        posterior cruciate and the lateral collateral, neither of which is a target here,
        and a positive outranks every negative in the scorer - so one PCL clause was
        enough to override an explicit "the ACL is normal".

        The test is proximity to the structure's own stem, not presence in the clause.
        "Posterior horn of the medial meniscus" appears in a large share of knee reports
        and says nothing about a cruciate; only a qualifier sitting beside the ligament
        word is one. A clause naming both sides keeps the match, because it does mention
        this target.
        """
        if self.contrary is None or self.stem is None:
            return False
        other = _distance(clause, self.stem, self.contrary, self.window)
        if other is None:
            return False
        own = _distance(clause, self.stem, self.side, self.window)
        # Not "is the other side mentioned" but "is it the nearer of the two". A clause
        # reading "tear of the posterior horn of the medial meniscus; the cruciate
        # ligaments are intact" mentions posterior, and under a presence test that was
        # enough to suppress the cruciate cue - which is the opposite of the intent,
        # since the clause does describe the ligaments. Ties go to keeping the match:
        # a clause naming both sides does mention this one.
        return own is None or other < own


# Which cue, if it sits beside the structure's stem, means the clause is about the other
# member of the pair. Only the two structures with a side-blind cue need one.
CONTRARY = {"ACL": SIDE_POSTERIOR, "MCL": SIDE_LATERAL}

ANAT_MATCH = {
    tgt: _Matcher(ANAT[tgt], *STEM_RULES[tgt], contrary=CONTRARY.get(tgt))
    for tgt in PAIRED
}
COMPARTMENT_MATCH = {
    "Medial OA": _Matcher(COMPARTMENT["Medial OA"], *STEM_RULES["Medial OA"]),
    "Lateral OA": _Matcher(COMPARTMENT["Lateral OA"], *STEM_RULES["Lateral OA"]),
    "PF OA": _Matcher(COMPARTMENT["PF OA"]),
}
DIRECT_MATCH = {
    tgt: _Matcher(_rx(rx.pattern, STEM_FRACTURE.pattern) if tgt == "Fracture" else rx)
    for tgt, rx in DIRECT.items()
}


def _severity(clause: str) -> float:
    """Weight one positive mention by how emphatic the sentence is.

    Ordered, not calibrated. A "moderate effusion" must outrank a "trace effusion" and
    both must outrank silence; the absolute numbers do not matter to AUC.
    """
    high = SEV_HIGH.search(clause) is not None
    low = SEV_LOW.search(clause) is not None
    if high and not low:
        return 1.0
    if low and not high:
        return 0.45
    return 0.75                       # unqualified mention


def _score_clauses(cls, anat_rx, path_rx=None, decoy_rx=None, context_penalty=None,
                   context_bonus=None):
    """Accumulate graded evidence over clauses for one target.

    Returns (score, confidence, n_pos, n_neg). Positives are graded by severity and by
    optional context regexes; negatives only matter when nothing positive was found,
    because reports assert normality for every structure they check.
    """
    n_pos = n_neg = n_unc = 0
    best = 0.0
    for c in cls:
        m = anat_rx.search(c)
        if not m:
            continue
        if decoy_rx is not None and decoy_rx.search(c):
            continue
        if path_rx is not None and not path_rx.search(c):
            if NORMALITY.search(c) and not NEGATION.search(c):
                n_neg += 1
            continue
        pol = _polarity(c, m.end())
        if pol == "positive":
            n_pos += 1
            w = _severity(c)
            if context_penalty is not None and context_penalty.search(c):
                w *= 0.45
            if context_bonus is not None and context_bonus.search(c):
                w = min(1.0, w * 1.35)
            best = max(best, w)
        elif pol == "negative":
            n_neg += 1
        else:
            n_unc += 1
            best = max(best, 0.30)

    if n_pos or n_unc:
        # 0.52 .. 0.95, ordered by the strongest single mention, nudged by repetition.
        score = min(0.95, 0.50 + 0.42 * best + 0.03 * min(n_pos, 3))
        conf = min(1.0, 0.55 + 0.15 * n_pos)
    elif n_neg:
        score = max(0.04, 0.20 - 0.04 * n_neg)
        conf = min(0.9, 0.45 + 0.12 * n_neg)
    else:
        score, conf = 0.28, 0.05          # silence sits above asserted-negative
    return score, conf, n_pos, n_neg


def extract(report: str) -> dict:
    """Extract twelve (score, confidence) pairs from one report."""
    cls = clauses(report)
    out = {}
    path_paired = _rx(TEAR.pattern, DEGEN.pattern, INJURY.pattern)

    for tgt in TARGETS:
        if tgt in PAIRED:
            s, c, npos, nneg = _score_clauses(cls, ANAT_MATCH[tgt], path_paired)
        elif tgt in OA_TARGETS:
            s, c, npos, nneg = _score_clauses(cls, COMPARTMENT_MATCH[tgt], OA_EVIDENCE)
        elif tgt == "Contusion":
            # Reactive subchondral oedema under a cartilage defect is osteoarthritis,
            # not a bruise. Explicit trauma wording pushes the other way.
            s, c, npos, nneg = _score_clauses(cls, DIRECT_MATCH[tgt], None, DECOY.get(tgt),
                                              context_penalty=DEGENERATIVE_MARROW,
                                              context_bonus=TRAUMA)
        else:
            s, c, npos, nneg = _score_clauses(cls, DIRECT_MATCH[tgt], None, DECOY.get(tgt))
        out[tgt] = s
        out[tgt + "__conf"] = c
        out[tgt + "__npos"] = npos
        out[tgt + "__nneg"] = nneg

    # --- cross-target corrections ------------------------------------------ #
    # A whole-joint osteoarthritis statement is evidence for every compartment that was
    # not separately assessed. Without this, "incipient OA of all three compartments"
    # scores zero on all three OA targets.
    g_hits = [c for c in cls if GLOBAL_OA.search(c) and _polarity(c, 0) == "positive"]
    if g_hits:
        gscore = 0.50 + 0.42 * max(_severity(c) for c in g_hits)
        for tgt in OA_TARGETS:
            if out[tgt + "__npos"] == 0 and out[tgt + "__nneg"] == 0:
                out[tgt] = max(out[tgt], gscore * 0.92)
                out[tgt + "__conf"] = max(out[tgt + "__conf"], 0.4)

    # Synovitis is frequently visible on the images and absent from the text, so silence
    # is weak evidence of absence here in a way it is not for other findings. Effusion is
    # its most reliable textual proxy - the two share a mechanism - so a silent synovitis
    # inherits a fraction of the effusion evidence instead of falling to the floor.
    if out["Synovitis__npos"] == 0 and out["Synovitis__nneg"] == 0:
        out["Synovitis"] = max(out["Synovitis"], 0.28 + 0.45 * (out["Effusion"] - 0.28))

    return out



# %% [markdown]
# ## 3. Reading the acquisition
# 
# `train_series.csv` describes each series with an anatomical plane and two binary flags,
# `Fluid_Sensitive` and `Fat_Suppression`. The names denote two physically independent
# properties — and that independence is what the delivered columns do not have: across the
# training series the two agree on every row, so as given they carry one axis rather than
# two. That is the first reason to recover both from the header.
# 
# *Fluid sensitivity* is a property of the **contrast weighting**, set by repetition time
# $T_R$ and echo time $T_E$:
# 
# $$
# \text{weighting} \;=\;
# \begin{cases}
# T_1 & T_R \lesssim 800\ \text{ms}\\
# T_2 & T_R \gtrsim 800\ \text{ms},\ T_E \gtrsim 60\ \text{ms}\\
# \text{PD} & T_R \gtrsim 800\ \text{ms},\ T_E \lesssim 60\ \text{ms}
# \end{cases}
# $$
# 
# Fluid is bright on $T_2$, intermediate on proton density, dark on $T_1$. Gradient echo
# breaks the rule — its $T_R$ is short by design — so it is settled by `ScanningSequence`
# first. *Fat suppression* is a **preparation** applied on top of any weighting, and is what
# makes marrow oedema conspicuous. Both are read from `SeriesDescription`, `SequenceName` and
# `ScanOptions` where the protocol names them, and from $T_R$ and $T_E$ where it does not.
# 
# ### Which sequences to show the model
# 
# A knee is read in three planes because the structures run in different directions:
# cruciate ligaments obliquely, best seen sagittally; collateral ligaments and the meniscal
# body coronally; patellar cartilage and the retinacula axially. Crossing plane with the two
# acquisition axes gives the slots below, chosen so each of the twelve findings has at least
# one sequence that shows it well.
# 
# | slot | plane | weighting | fat sat | what it carries |
# |---|---|---|---|---|
# | `SAG_FLUID_FS` | sagittal | PD / T2 | yes | meniscal tears, marrow oedema, effusion |
# | `COR_FLUID_FS` | coronal | PD / T2 | yes | collateral ligaments, meniscal body, oedema |
# | `AX_FLUID_FS` | axial | PD / T2 | yes | patellofemoral joint, synovium, effusion |
# | `SAG_FLUID_NOFS` | sagittal | PD / T2 | no | meniscal morphology at high contrast-to-noise |
# | `COR_T1` | coronal | T1 | no | marrow architecture, cartilage and bone outline |
# | `SAG_T1` | sagittal | T1 | no | anatomy, chronic change |
# 
# A study rarely has all six; a per-slot presence mask carries the absences into the head,
# which §6 uses.
# 


# %% cell 13
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import gc
import hashlib
import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F

# The label extractor is defined in the cells above when this runs as a notebook. As a
# plain script it is imported from the package source, so the two paths share one
# definition rather than keeping a copy each.

T0 = time.time()
SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]


# The centre crop has to be smaller than the smallest field of view in the corpus or it
# silently does nothing. Measured over every training series, the acquired field of view
# (Rows x PixelSpacing) has median 160 mm and runs from 70 to 320: a 160 mm crop is
# larger than the image in 60% of series and is skipped for all of them, which leaves
# their physical scale unnormalised. 130 mm is below the field of view of 99.6% of
# series and still contains the joint.
CROP_MM = 130.0

# Cache resolution. Everything downstream may downsample from this, so it is set by the
# most demanding configuration rather than by the default one.
CACHE_IMG = 336
GROUP = 3                  # slices per encoder input, stacked as the three channels
N_GROUP_MAX = 1
CACHE_FRACTION = 0.45      # share of free memory the pixel cache may take
CACHE_BUDGET_MAX_GB = 24.0 # hard ceiling regardless of what the machine reports
CACHE_BUDGET_GB = 12.0     # only the fallback, for a machine with no /proc/meminfo
TEST_SHARE = 0.30          # floor on the test corpus relative to the training one, since
                           # the visible test split is a stub and the scored one is not
HDR_THREADS = 16
PIX_THREADS = 12
ORDER_THREADS = 32         # slice-ordering is latency-bound on the mount, not CPU-bound
# Ceiling for the ordering pass. It has to be a ceiling because the pass is hundreds of
# thousands of small reads over a network mount, so its duration is a property of the
# mount rather than of the work, and varies between runs that do the same reading. It must not be a tight one: giving up leaves
# those series in file order, which is uncorrelated with anatomy, and that degradation is
# silent. So the ceiling sits well above what the pass ordinarily needs: its purpose is
# to stop the pass consuming the whole run on a slow mount, not to trim the ordinary
# case, and a ceiling tight enough to bind on a normal day would trade a silent
# degradation for a saving the run does not need.
ORDER_BUDGET_S = 5400

# Resolution is the axis under test. A feature of width d mm survives resampling only if
# the pixel pitch is at most d/2, and the pitch here is set by the crop above rather than
# by the acquired field of view: CROP_MM / P. At 224 px that is 0.58 mm, above the 0.5 mm
# a 1 mm tear needs; at 336 px it is 0.39 mm and clears it. Both configurations read the
# same cache, so the comparison isolates the resize.
RUNS = [
    {"name": "r224", "img": 224},
    {"name": "r336", "img": 336},
]

EPOCHS = 10
BATCH_STUDIES = 8          # a study is a bag of up to N_SLOT slot images
AUG_ROT_DEG = 8.0          # rigid jitter; see augment() for why neither flip is used
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.10
LAT_MIN_OFFSET_MM = 20.0   # inside this the side is not readable from geometry; see
                           # side_from_geometry()
SLICE_BAND = (0.20, 0.80)  # fraction of the ordered stack read_slot samples across

# --- What a slice IS, as opposed to how many of them there are --------------- #
#
# A member is a function of the pixels it was fitted on, and img/crop_mm/slices/band do
# not determine those pixels by themselves. Four further decisions do, none of them
# visible in any shape:
#
#   order          which slice is the next one along the stack
#   lat            which knees are mirrored, and on what evidence
#   slot_fallback  whether a T1 slot may be filled from a series that is not T1
#   decode_fill    what stands in for a slice that would not decode
#
# `native` is the reading derived in the sections below. `legacy` is the reading an
# imported member was fitted under. A member read under the wrong one loads with every
# shape matching, runs, and writes a plausible submission computed from the wrong image -
# so the choice travels with the member and is part of the key that decides which members
# can share a decode. The legacy rules are reproduced rather than corrected: correcting
# them would hand that member pixels its weights never saw.
RULES_NATIVE = {"order": "normal", "lat": "centre",
                "slot_fallback": False, "decode_fill": "nearest"}
RULES_LEGACY = {"order": "dominant_axis", "lat": "corner_x",
                "slot_fallback": True, "decode_fill": "zero"}
RULES = dict(RULES_NATIVE)
LEGACY_LAT_OFFSET_MM = 5.0   # the dead zone the legacy laterality rule was fitted with

LR_HEAD = 1e-3
LR_BACKBONE = 8e-6         # the encoder is adapted, not retrained
UNFREEZE_LAST = 6          # trainable transformer blocks, from the output end
WEIGHT_DECAY = 0.02
EVAL_BATCH = 8
TIME_BUDGET = 8.0 * 3600

# Six slots: three planes crossed with the acquisition axes. The fat-suppressed
# fluid-sensitive series exist for nearly every study; the T1 and the non-suppressed
# fluid-sensitive series are scarcer, which is what the presence mask is for.
SLOTS_RECOVERED = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]

# The alternative: plane x the single axis the delivered flags carry, ignoring the
# recovered weighting. Kept as
# a switch so the choice of slot definition can be varied while everything else is held
# fixed. Under this scheme a `Struct` slot mixes T1 series with non-fat-suppressed PD/T2
# series, which carry very different tissue contrast.
SLOTS_PUBLIC = [
    ("SAG_FLUID", "Sagittal", None, True),
    ("COR_FLUID", "Coronal", None, True),
    ("AX_FLUID", "Axial", None, True),
    ("SAG_STRUCT", "Sagittal", None, False),
    ("COR_STRUCT", "Coronal", None, False),
    ("AX_STRUCT", "Axial", None, False),
]

SLOT_SCHEME = os.environ.get("SLOT_SCHEME", "recovered")
SLOTS = SLOTS_PUBLIC if SLOT_SCHEME == "public" else SLOTS_RECOVERED
N_SLOT = len(SLOTS)

# How many 384-wide parts the per-slot feature is built from. The encoder emits one
# vector per token; a slot feature is a fixed summary of that grid, and the summary an
# imported member was fitted with carries a third part.
POOL_PARTS = {"cls_mean": 2, "cls_mean_focal": 3}

# Which slots an imported member's attention is tilted toward, per diagnosis. Indices are
# into SLOTS. This is a fixed table rather than a learned parameter, so it is part of that
# member's definition and has to be reproduced exactly for its weights to mean anything.
SLOT_PRIOR_TABLE = {
    "ACL": (0, 3, 5), "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4), "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5), "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5), "Effusion": (0, 2), "Synovitis": (0, 2),
    "Baker's": (0,), "Contusion": (0, 1, 2), "Fracture": (0, 1, 2, 4, 5),
}
SLOT_PRIOR_STRENGTH = 0.55

FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
_SEP = re.compile(r"[_\-.]")
_FATSAT_RX = re.compile(r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
                        r"water excit|\btirm\b|\bsting\b|\bfatsup\b")
_T1_RX = re.compile(r"\bt1\b|\bt1w\b")
_T2_RX = re.compile(r"\bt2\b|\bt2w\b")
_PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")



# %% cell 14
def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find_root():
    for c in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
              Path("/kaggle/input/rsna-knee-abnormality-detection"),
              Path("data"), Path(".")]:
        if (c / "test.csv").is_file() and (c / "test_series").is_dir():
            return c
    # last resort: two-level scan, because the mount is nested one deeper than usual
    base = Path("/kaggle/input")
    if base.is_dir():
        for depth1 in sorted(p for p in base.iterdir() if p.is_dir()):
            for cand in [depth1] + sorted(p for p in depth1.iterdir() if p.is_dir()):
                if (cand / "test.csv").is_file():
                    return cand
    raise FileNotFoundError(
        f"competition mount not found (cwd {Path.cwd()}); expected a directory holding "
        f"test.csv and test_series/")


def find_dinov2(variant="small"):
    """Locate a mounted DINOv2 checkpoint directory by variant name."""
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    hits = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if "config.json" in files and "dinov2" in root.lower():
            hits.append(Path(root))
    for h in hits:
        if variant in str(h).lower():
            return h
    return hits[0] if hits else None


LABEL_COLS = TARGETS + [t + "__conf" for t in TARGETS]


class LabelSourceError(RuntimeError):
    """Raised when the labels did not come from where this run intended.

    Every other failure in this file is better survived than reported: a run that dies
    after the cache is built has spent the expensive half and scores nothing, so the
    guard around `main` swallows it and leaves the benchmark file behind. This one is
    the exception. Training on the weaker labels does not look like a failure - it
    completes, writes a plausible submission, and differs only in a log line - so it has
    to stop the run rather than be absorbed by a guard designed for crashes.
    """


def find_label_table():
    """Locate a mounted table of pre-read report labels, if one is attached.

    The lexicon turns a report into labels by matching morphology, and its failure
    mode is silence: on a phrasing it does not carry it emits no opinion rather than a
    wrong one. Silence is measurable without any ground truth - for each (report,
    finding) pair, did anything match? - and that measurement says the misses are
    concentrated in particular languages rather than spread evenly, on findings a knee
    report almost always comments on.

    Enumerating morphology for nine languages is the wrong instrument for that. Reading
    the sentence is the right one, and a language model reads it. Against the annotated
    studies the difference is large and one-sided, so when such a table is mounted it is
    preferred; when it is not, the lexicon runs and the pipeline is unchanged. Both paths
    produce the same columns, so nothing downstream knows which one supplied them.
    """
    base = Path("/kaggle/input")
    cands = []
    if base.is_dir():
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
            cands += [Path(root) / f for f in files if f.startswith("report_labels")
                      and f.endswith(".csv")]
    cands += [p for p in (Path("data/derived/report_labels_v2.csv"),) if p.is_file()]
    for c in cands:
        try:
            head = pd.read_csv(c, nrows=1)
        except Exception:
            continue
        if "StudyInstanceUID" in head.columns and all(t in head.columns for t in TARGETS):
            return c
    return None


def label_mount_attached():
    """True when an input directory was attached that is meant to carry a label table.

    The fallback below is deliberate and has to stay silent for a run with no table
    attached, because that is the ordinary case for anyone reading this notebook. It
    must not stay silent for the other case: a table was attached and could not be used.
    Those two are indistinguishable from the labels alone - both end with the lexicon -
    so they are separated here by whether the mount exists at all.
    """
    base = Path("/kaggle/input")
    if not base.is_dir():
        return False
    return any("label" in p.name.lower() for p in base.iterdir() if p.is_dir())


def read_labels(train_df):
    """Labels for every training study, from a mounted table or from the lexicon.

    Studies the mounted table does not cover fall back to the lexicon rather than being
    dropped, so a partial table degrades coverage instead of losing rows.
    """
    n = len(train_df)
    lab = pd.DataFrame([extract(r) for r in train_df["Report"].fillna("")])
    lab["StudyInstanceUID"] = train_df["StudyInstanceUID"].values
    lab = lab.set_index("StudyInstanceUID")

    src = find_label_table()
    if src is None:
        if label_mount_attached():
            raise LabelSourceError(
                "LABEL SOURCE: a label dataset is mounted but no usable table was found "
                "in it. Falling back to the lexicon here would train on the weaker "
                "labels and say so only in a log line, so the run stops instead.")
        log(f"LABEL SOURCE: lexicon, {n} studies (no table mounted)")
        return lab

    tab = pd.read_csv(src).set_index("StudyInstanceUID")
    missing = [c for c in LABEL_COLS if c not in tab.columns]
    if missing:
        raise LabelSourceError(
            f"LABEL SOURCE: {src} is missing {len(missing)} expected columns "
            f"(first: {missing[0]!r}). Refusing to fall back silently.")
    hit = lab.index.intersection(tab.index)
    if not len(hit):
        raise LabelSourceError(
            f"LABEL SOURCE: {src} shares no StudyInstanceUID with train.csv.")
    log(f"LABEL SOURCE: {src.name} covers {len(hit)} of {n} studies, "
        f"lexicon for the remaining {n - len(hit)}")
    lab.loc[hit, LABEL_COLS] = tab.loc[hit, LABEL_COLS].values
    return lab


ROOT = find_root()
log(f"input root: {ROOT}")


IMG = CACHE_IMG            # kept as the name the pixel reader and cache use


def available_gb():
    """Memory this machine will actually lend, read rather than assumed.

    A hardcoded ceiling is a guess about a machine the author is not sitting at, and a
    guess that is too low costs coverage silently while a guess that is too high ends the
    run. The machine will say, so it is asked.
    """
    try:
        with open("/proc/meminfo") as fh:
            info = {k.strip(): v for k, v in
                    (l.split(":", 1) for l in fh if ":" in l)}
        return int(info["MemAvailable"].split()[0]) / 1024 ** 2
    except Exception:
        return CACHE_BUDGET_GB / CACHE_FRACTION      # fall back to the old constant


def plan_cache(n_study, n_test=0):
    """Choose how many slices per slot the memory the machine has will allow.

    The cache is n_study x n_slot x slices x IMG^2 bytes. Coverage is the cheap axis -
    linear - and resolution the expensive one, so when the budget binds it is the slice
    count that gives way rather than the pixel grid. Deciding once, from the training
    corpus size, keeps train and test caches on the same group layout.

    Only a fraction of what is free is taken. The rest is not slack: the encoder, its
    activations, the pinned batches and the frames all come out of the same pool, and the
    cache is the one allocation big enough that overshooting it kills the run outright.
    """
    avail = available_gb()
    budget = min(avail * CACHE_FRACTION, CACHE_BUDGET_MAX_GB)
    # Both caches are held at once, and the test half is what the visible run cannot
    # show: here it is a handful of studies, and at scoring it is the whole hidden set.
    # Sizing against the training corpus alone therefore passes every run that can be
    # watched and overruns the one that counts.
    n_total = n_study + max(n_test, int(TEST_SHARE * n_study))
    per_slice = n_total * N_SLOT * IMG * IMG
    afford = int(budget * 1024 ** 3 // max(per_slice, 1))
    groups = max(1, min(N_GROUP_MAX, afford // GROUP))
    log(f"memory: {avail:.1f} GB available, {budget:.1f} GB to the cache; "
        f"sizing for {n_study} train + {n_total - n_study} test studies "
        f"-> {groups} group(s) of {GROUP} = {groups * GROUP} slices per slot"
        + (f" (wanted {N_GROUP_MAX})" if groups < N_GROUP_MAX else ""))
    return groups


N_GROUP = plan_cache(len(pd.read_csv(ROOT / "train.csv")),
                     len(pd.read_csv(ROOT / "test.csv")))
CACHE_SLICES = GROUP * N_GROUP
log(f"cache layout: {N_GROUP} groups x {GROUP} slices = {CACHE_SLICES} per slot")



# %% cell 15
HDR_TAGS = ["SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
            "RepetitionTime", "EchoTime", "Laterality", "PixelSpacing", "Rows",
            "Columns", "RescaleSlope", "RescaleIntercept",
            # Position and orientation are read from the same header probe() already
            # opens, so they cost nothing, and they are what recovers the side when the
            # Laterality tag is absent - which it is for half the studies here.
            "ImagePositionPatient", "ImageOrientationPatient"]


def _hdr_vec(s, n):
    """Parse a DICOM multi-value string as stored by probe(): floats joined by `|`."""
    if not isinstance(s, str):
        return None
    try:
        v = [float(x) for x in s.split("|")]
    except ValueError:
        return None
    return np.array(v) if len(v) >= n else None


def side_from_geometry(h):
    """Study -> 'L' / 'R' / None, from where the image sits in the patient.

    `Laterality` (0020,0060) is Type 2C and may legitimately be absent; in this corpus it
    is missing on exactly half the studies, and the vendors it is missing from are whole
    vendors rather than scattered series. A study with no tag is not a left knee, but the
    normalisation upstream treats it as one, so half the corpus was never normalised and
    the five side-defined targets - the two menisci, the two tibiofemoral compartments
    and the medial collateral ligament - saw that axis reversed on a large minority of it.

    The patient coordinate system fixes this without the tag: +x is the patient's left, so
    the centre of a right knee sits at negative x. The centre is used rather than
    `ImagePositionPatient` itself because that is the corner of the image, which is offset
    by half a field of view - enough to change the sign on a knee near the midline.

    The median over a study's series is what is thresholded, not a single series: probe()
    reads one arbitrary slice per series, which on a sagittal stack can sit anywhere
    across the joint. Studies whose centre falls near the midline are left unresolved
    rather than guessed - measured against the tagged half, the rule is right 97% of the
    time overall and no better than chance inside 20 mm.
    """
    cx = {}
    for r in h.itertuples(index=False):
        ipp = _hdr_vec(getattr(r, "ImagePositionPatient", None), 3)
        iop = _hdr_vec(getattr(r, "ImageOrientationPatient", None), 6)
        ps = _hdr_vec(getattr(r, "PixelSpacing", None), 2)
        rows, cols = getattr(r, "Rows", None), getattr(r, "Columns", None)
        if ipp is None or iop is None or ps is None or not rows or not cols:
            continue
        try:
            c = ipp[:3] + iop[:3] * ps[1] * float(cols) / 2 + iop[3:6] * ps[0] * float(rows) / 2
        except (TypeError, ValueError):
            continue
        cx.setdefault(r.StudyInstanceUID, []).append(float(c[0]))
    out = {}
    for st, xs in cx.items():
        m = float(np.median(xs))
        out[st] = None if abs(m) < LAT_MIN_OFFSET_MM else ("R" if m < 0 else "L")
    return out


def side_from_corner_x(h):
    """The laterality an imported member was fitted under.

    It thresholds the median raw `ImagePositionPatient` x over a study's series. That is
    the x of the image *corner*, not of its centre, so it differs from the rule above by
    up to half a field of view - which is enough to reverse the sign on a knee scanned
    near the midline. The dead zone is 5 mm rather than 20 mm, so it also commits on
    studies the rule above leaves unresolved.

    Neither difference changes a shape. Each one decides whether a study is mirrored, and
    a study mirrored one way at training and the other at inference presents the five
    side-defined targets with their axis reversed.
    """
    out = {}
    for st, g in h.groupby("StudyInstanceUID"):
        xs = []
        for r in g.itertuples(index=False):
            ipp = _hdr_vec(getattr(r, "ImagePositionPatient", None), 3)
            if ipp is not None and np.isfinite(ipp).all():
                xs.append(float(ipp[0]))
        if not xs:
            out[st] = None
            continue
        x = float(np.median(xs))
        # DICOM patient coordinates are LPS: +x is the patient's left.
        out[st] = None if abs(x) < LEGACY_LAT_OFFSET_MM else ("R" if x < 0 else "L")
    return out


def lat_of(h, tag=""):
    """Study -> 'L' / 'R' / None: the tag where it exists, geometry where it does not.

    The tag is present on exactly half the studies here and is sometimes an empty
    string rather than absent, which is not the same as NaN. Treating the other half
    as left-sided is what `normalise_laterality` did by omission, so the geometry
    fallback is not a refinement - it is the difference between normalising half the
    corpus and normalising all of it.
    """
    geo = side_from_corner_x(h) if RULES["lat"] == "corner_x" else side_from_geometry(h)
    d, n_tag, n_geo, n_none, n_disagree = {}, 0, 0, 0, 0
    for st, g in h.groupby("StudyInstanceUID"):
        v = [str(x).strip().upper() for x in g["Laterality"].dropna()]
        if RULES["lat"] == "corner_x" and "ImageLaterality" in g.columns:
            # The legacy rule reads the second tag too, so a study tagged only there is
            # resolved from the tag rather than from geometry.
            v += [str(x).strip().upper() for x in g["ImageLaterality"].dropna()]
        v = [x[0] for x in v if x and x[0] in ("L", "R")]
        side = v[0] if v else None
        if side is not None:
            n_tag += 1
            if geo.get(st) is not None and geo[st] != side:
                n_disagree += 1
        else:
            side = geo.get(st)
            n_geo += side is not None
            n_none += side is None
        d[st] = side
    log(f"{tag}laterality: {n_tag} from the tag, {n_geo} from geometry, "
        f"{n_none} unresolved; tag and geometry disagree on {n_disagree} "
        f"({n_disagree / max(n_tag, 1):.1%} of the tagged)")
    return d



def probe(item):
    split, study, series, path = item
    row = {"split": split, "StudyInstanceUID": study, "SeriesInstanceUID": series,
           "dir": path}
    try:
        files = sorted(e.name for e in os.scandir(path) if e.name.endswith(".dcm"))
        row["files"] = files
        row["n_slices"] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(os.path.join(path, files[len(files) // 2]),
                             stop_before_pixels=True, force=True)
        for t in HDR_TAGS:
            v = getattr(ds, t, None)
            if v is None:
                row[t] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
                row[t] = "|".join(str(x) for x in v)
            else:
                row[t] = str(v)
    except Exception as exc:
        row["err"] = str(exc)[:120]
    return row


def walk(split):
    """Every series directory of a split, with one header read per series.

    An absent split returns an empty frame *with the columns annotate expects*. Returning
    a bare DataFrame looks like the same thing and is not: the next call indexes
    `SeriesDescription` and raises KeyError, so the branch that exists to survive a
    missing split is what turns it into a crash.
    """
    base = ROOT / split
    items = []
    if not base.is_dir():
        return pd.DataFrame(columns=["split", "StudyInstanceUID", "SeriesInstanceUID",
                                     "dir", "files", "n_slices"] + HDR_TAGS)
    for study in os.scandir(base):
        if study.is_dir():
            for series in os.scandir(study.path):
                if series.is_dir():
                    items.append((split, study.name, series.name, series.path))
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as pool:
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)


def annotate(df):
    """Recover fat suppression and pulse-sequence weighting from the header."""
    desc = (df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna(""))
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)

    opts = df["ScanOptions"].fillna("").str.upper().str.split("|")
    # GE writes SAT_GEMS for spatial saturation, so ScanOptions must be matched as
    # exact tokens; a substring test on "SAT" fires on non-fat-sat series.
    opts_fs = opts.apply(lambda ts: any(t.strip() in FATSAT_OPTS for t in ts))
    df["fatsat"] = desc.str.contains(_FATSAT_RX) | opts_fs

    tr = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    te = pd.to_numeric(df["EchoTime"], errors="coerce")
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    t1, t2, pdw = desc.str.contains(_T1_RX), desc.str.contains(_T2_RX), desc.str.contains(_PD_RX)

    df["weight"] = np.where(t1 & ~t2 & ~pdw, "T1",
                     np.where(t2 & ~pdw, "T2",
                       np.where(pdw, "PD",
                         np.where(gre, "GRE",
                           np.where(tr < 800, "T1",
                             np.where(te > 60, "T2",
                               np.where(tr >= 800, "PD", "UNK")))))))
    df["fluid"] = np.isin(df["weight"], ["PD", "T2"])
    df["px"] = pd.to_numeric(
        df["PixelSpacing"].fillna("").str.split("|").str[0].replace("", np.nan),
        errors="coerce")
    return df



# %% cell 16
def pick_slots(series_df, plane_map):
    """One series per slot per study.

    Ties are broken toward the stack with the most slices: a thicker stack samples the
    joint more densely, and the three-slice sampler below benefits from the margin.
    """
    series_df = series_df.copy()
    series_df["plane"] = series_df["SeriesInstanceUID"].map(plane_map)
    out = {}
    for study, g in series_df.groupby("StudyInstanceUID"):
        chosen = {}
        for name, plane, fluid, fs in SLOTS:
            sel = (g["plane"] == plane) & (g["fatsat"] == fs)
            # fluid=None means "do not condition on weighting" - the public scheme,
            # where the single provided flag stands in for both axes at once.
            if fluid is not None:
                sel &= (g["fluid"] == fluid)
            cand = g[sel]
            # A slot with no series matching its predicate stays empty, and no substitute
            # is admitted from a neighbouring predicate. Relaxing the weighting to fill a
            # T1 slot would draw from the pool `SAG_FLUID_NOFS` selects from, since that
            # pool is what remains once the weighting is dropped: over the training corpus
            # it would put one series in two slots for 2383 of 4407 studies and leave 56%
            # of the T1 slot holding PD or T2. The presence mask would then assert a
            # sequence that was never acquired, and the per-diagnosis softmax of §6 would
            # divide its attention across two identical slots, giving one acquisition
            # about twice the weight it carries in a study that holds both. The mask is
            # there to say a slot is absent, which is what an absent slot is.
            if len(cand) == 0 and RULES["slot_fallback"] and fluid is False:
                # The relaxation the paragraph above rejects, reproduced because an
                # imported member was fitted with its T1 slots filled this way: over half
                # of that member's training studies had a T1 slot holding a series that
                # is not T1. Leaving those slots empty would present it with a presence
                # mask it never saw.
                cand = g[(g["plane"] == plane) & (~g["fatsat"])]
            if len(cand):
                chosen[name] = cand.sort_values("n_slices", ascending=False).iloc[0]
        out[study] = chosen
    return out



# %% [markdown]
# ## 3b. What order the slices are in
# 
# A series is a directory of files, and the obvious way to walk it is to sort the file names.
# That is wrong here, and wrong in a way that produces no error. The file name is the SOP
# Instance UID, assigned to be unique rather than ordered, so sorting by it yields a sequence
# uncorrelated with anatomy — measured on a series from this corpus, the rank correlation
# between file-name order and physical position is $\rho \approx 0.01$. Three things
# downstream depend on that order: three adjacent slices as three channels become three
# unrelated cross-sections; the middle of the stack becomes a random subset; and reversing
# the order to normalise laterality does nothing at all.
# 
# The true order is recoverable exactly from geometry every slice carries.
# `ImageOrientationPatient` gives the in-plane axes $\hat{r}_x, \hat{r}_y$ and
# `ImagePositionPatient` the position $p$ of the first voxel, so
# 
# $$\hat{n} \;=\; \hat{r}_x \times \hat{r}_y, \qquad k \;=\; p \cdot \hat{n},$$
# 
# and $k$ increases monotonically along the stack. Because $k$ is signed and expressed in
# patient coordinates, the stack has a fixed direction along the body's left-right axis —
# which is what the laterality normalisation of §5 reverses. `InstanceNumber` is the fallback
# where the geometry tags are missing; it usually tracks $k$ up to sign, but interleaved and
# multi-echo acquisitions number slices in an order that is not the order they occupy in
# space, and it is not signed in patient coordinates.
# 


# %% [markdown]
# ## 4. Sampling: how many millimetres one pixel is allowed to be
# 
# A DICOM slice of $N \times N$ pixels with spacing $s$ mm/pixel covers $Ns$ millimetres of
# anatomy. Both vary across this corpus, so a fixed-pixel resize hands the encoder images
# whose physical scale differs by a factor of several. But scale normalisation is only half
# of it; the other half is a hard limit.
# 
# **A feature narrower than two pixels does not survive the resize.** To represent a
# structure of width $d$ millimetres the pixel pitch must satisfy $s_{\text{eff}} \le d/2$,
# the Nyquist condition applied to the resampling grid. A meniscal tear is one to three
# millimetres, so at $d = 1$ mm the pitch must be at most $0.5$ mm — and if it is not, no
# capacity downstream recovers the signal, because it was destroyed before the first
# convolution. This is a property of the resize, not of the network.
# 
# Cropping to a constant physical extent $L$ and resampling to $P$ pixels fixes the pitch:
# 
# $$n \;=\; \Big\lfloor \frac{L}{s} \Big\rceil \ \text{pixels}, \qquad
# s_{\text{eff}} \;=\; \frac{L}{P}\ \ \text{mm/pixel}, \qquad
# \text{token} \;=\; 14\,s_{\text{eff}}\ \ \text{mm}.$$
# 
# The crop must be smaller than the smallest field of view or it silently does nothing: if
# $L/s$ exceeds the image width the crop cannot be taken and that series passes through
# unnormalised, with no error. $L = 130$ mm is below the acquired field of view of almost
# every series here while still containing the joint. The resize target then follows from the
# tear width rather than convention — at $L = 130$ mm an input of $224$ gives $0.580$
# mm/pixel, above the bound for a 1 mm feature, while $336$ gives $0.387$ mm/pixel and puts a
# $14$-pixel patch token at $5.4$ mm.
# 
# **Intensity needs the same treatment**, MR having no absolute scale. Each series is
# normalised to its own 1st and 99th percentile — over the sampled stack rather than per
# slice, so slices keep their relative contrast, and percentiles rather than extremes, so one
# bright vessel does not compress everything else.
# 


# %% cell 19
ORDER_TAGS = [(0x0020, 0x0032), (0x0020, 0x0037), (0x0020, 0x0013)]

# Series in which at least one sampled slice would not decode. A list rather than a
# counter because appending is atomic under the reader threads, and reported rather than
# swallowed: unreported, a decode failure is indistinguishable from a black knee.
DECODE_FAILED = []


def cache_tag(rules=None):
    """The name a decoded cache is stored under.

    It has to name everything that decides the pixels, not only their dimensions. Two
    configurations that agree on resolution, slice count, crop and band but disagree on
    how a slice is chosen produce different arrays of identical shape - so a tag built
    from the dimensions alone lets the second attach to the first one's file and train
    against pixels it never asked for, with nothing anywhere reporting a mismatch.

    A native reading keeps the plain name, so caches decoded before the rules existed
    stay valid; anything else earns a suffix.
    """
    r = dict(RULES if rules is None else rules)
    t = (f"{CACHE_IMG}px_{CACHE_SLICES}sl_{int(CROP_MM)}mm_"
         f"{SLICE_BAND[0]:.2f}-{SLICE_BAND[1]:.2f}")
    if {k: r.get(k, v) for k, v in RULES_NATIVE.items()} != RULES_NATIVE:
        t += "_" + hashlib.md5(json.dumps(r, sort_keys=True).encode()).hexdigest()[:6]
    return t


def _natural_key(name):
    return tuple(int(x) if x.isdigit() else x.lower()
                 for x in re.split(r"(\d+)", str(name)))


def _order_dominant_axis(rec):
    """The slice order an imported member was fitted under.

    It sorts on the raw patient coordinate along whichever axis varies most across the
    stack, rather than on the projection onto the slice normal. The two differ by a sign,
    not by a formula: measured over this corpus every sagittal series has a slice normal
    with n_x in [-1.00, -0.98], so p.n is the negative of the raw x this sorts on and the
    two stacks come out exactly reversed. Because the band sampler truncates rather than
    rounds, its nine indices are not symmetric about the middle, so nine slices drawn from
    a twenty-six slice stack under one order share two with the other.

    Missing geometry falls back to `InstanceNumber` and then to a natural sort of the file
    name, both at the same 80% threshold the imported pipeline used.
    """
    files, d = rec["files"], rec["dir"]
    rows = []
    for pos, f in enumerate(files):
        ipp = inst = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True,
                                 specific_tags=["ImagePositionPatient", "InstanceNumber"])
            raw = getattr(ds, "ImagePositionPatient", None)
            if raw is not None and len(raw) >= 3:
                c = np.asarray(raw[:3], dtype=np.float64)
                if np.isfinite(c).all():
                    ipp = c
            n = getattr(ds, "InstanceNumber", None)
            if n is not None:
                inst = float(n)
        except Exception:
            pass
        rows.append((f, ipp, inst, pos))

    placed = [r for r in rows if r[1] is not None]
    need = max(2, int(0.8 * len(rows)))
    if len(placed) >= need:
        xyz = np.stack([r[1] for r in placed])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        rows.sort(key=lambda r: (float(r[1][axis]) if r[1] is not None else spare,
                                 r[2] if r[2] is not None else float("inf"), r[3]))
    elif sum(r[2] is not None for r in rows) >= need:
        rows.sort(key=lambda r: (r[2] if r[2] is not None else float("inf"), r[3]))
    else:
        rows.sort(key=lambda r: _natural_key(r[0]))
    return [r[0] for r in rows], True


def order_slices(rec):
    """Return the series' files sorted along the through-plane axis.

    A DICOM file name here is a SOP Instance UID, which is assigned arbitrarily. Sorting
    by it therefore produces an order uncorrelated with anatomy - measured over one
    series, Spearman between file-name rank and physical position is 0.009, i.e. none.
    Anything that assumes the file order means something is then operating on noise: the
    three channels of a "2.5D" input are three unrelated views rather than neighbouring
    slices, "the middle of the stack" is a random subset, and reversing slice order to
    normalise laterality reverses nothing meaningful.

    The physical order is recoverable exactly. Each slice carries its position in patient
    coordinates and the in-plane axes; projecting the position onto the slice normal
    gives a signed through-plane coordinate, monotonic along the stack:

        n = r_x  x  r_y ,      k = p . n

    `InstanceNumber` is the fallback. It usually tracks the projection up to sign, but
    interleaved and multi-echo acquisitions need not number slices in the order they
    occupy in space - but the projection is signed in patient
    coordinates, which is what laterality normalisation needs.
    """
    if RULES["order"] == "dominant_axis":
        return _order_dominant_axis(rec)
    files, d = rec["files"], rec["dir"]
    keyed = []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True,
                                 specific_tags=ORDER_TAGS)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                k = float(ds.InstanceNumber)
            except Exception:
                k = None
        keyed.append((k, f))
    if any(k is None for k, _ in keyed):
        # A series with no usable geometry keeps its arbitrary order; that is worse than
        # sorting but better than dropping the series, and it is logged as a count.
        return files, False
    return [f for _, f in sorted(keyed, key=lambda t: t[0])], True


def read_slot(rec, n_slice=None, out_size=None):
    """`n_slice` physically spread slices from one series, at `out_size` pixels.

    Returns uint8 [n_slice, out, out] normalised per-series to its 1st-99th
    percentile. Percentiles rather than min/max because MR intensity has no absolute
    scale and a single bright vessel would otherwise compress the whole dynamic range.

    Reading is the expensive half of this pipeline, so the caller reads once at the
    largest configuration it needs and derives the smaller ones from the returned buffer
    rather than re-reading.
    """
    n_slice = GROUP if n_slice is None else n_slice
    out_size = IMG if out_size is None else out_size
    files, d, px = rec.get("ordered") or rec["files"], rec["dir"], rec["px"]
    n = len(files)
    if n == 0:
        return None
    # Spread the samples over a central band of the stack: the outermost slices of a knee
    # series are mostly soft tissue outside the joint. The band is a constant rather than
    # a literal because how much of the stack is worth reading depends on how many slices
    # are being taken - at three the middle is all that fits, while at sixteen the ends
    # are worth having, and a Baker cyst sits at the posteromedial end of a sagittal one.
    lo, hi = int(SLICE_BAND[0] * (n - 1)), int(SLICE_BAND[1] * (n - 1))
    idx = np.unique(np.linspace(lo, hi, n_slice).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])

    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, "RescaleSlope", 1) or 1)
            ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
            a = a * sl + ic
        except Exception:
            a = None                      # no shape is known here; see below
        planes.append(a)

    # A slice that would not decode has no shape of its own, and inventing one is how a
    # single unreadable file erases a whole series: a substitute allocated at the resize
    # target while the decoded slices are still native makes the shape check below take
    # the substitute as the authority and zero the good slices with it, leaving a black
    # slot that the presence mask still reports as acquired.
    #
    # A failure is instead filled from the nearest slice that did decode - the same
    # convention the sampler already uses when the band holds fewer distinct slices than
    # were asked for - and a series where nothing decodes is reported absent, which the
    # mask can express, rather than black, which it cannot.
    got = [k for k, p in enumerate(planes) if p is not None]
    if RULES["decode_fill"] == "zero":
        # What an imported member was fitted with: a failure becomes a zero plane at the
        # resize target, which the shape check below then propagates to the whole slot.
        # It is the behaviour the paragraph above describes and rejects, kept here only
        # because that member's weights were learned against slots blacked out this way.
        if not got:
            DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
        planes = [np.zeros((out_size, out_size), np.float32) if p is None else p
                  for p in planes]
        got = list(range(len(planes)))
    if not got:
        DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
        return None
    if len(got) < len(planes):
        DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
        for k, p in enumerate(planes):
            if p is None:
                planes[k] = planes[min(got, key=lambda j: abs(j - k))]

    # Slices of one series can still differ in matrix size - multi-echo and some
    # reformats do - and those are genuinely not stackable.
    shp = planes[0].shape
    planes = [p if p.shape == shp else np.zeros(shp, np.float32) for p in planes]
    vol = np.stack(planes)

    # constant physical extent, then resize: PixelSpacing varies 3.4x across the corpus
    if px and np.isfinite(px) and px > 0:
        want = int(round(CROP_MM / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = h // 2, w // 2
            half = want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]

    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)

    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    # uint8, not float32. These buffers queue up between the reader threads and the
    # encoder, and at this size a float32 slot-series is several megabytes. Intensity is
    # already normalised into [0, 1] here, so eight bits cost nothing that a bilinear
    # resize has not already cost, and the queue is a quarter the size.
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)



# %% [markdown]
# ## 5. Normalising left and right
# 
# Four of the twelve targets — the two menisci and the medial and lateral tibiofemoral
# compartments — are medial/lateral pairs, and a fifth, the medial collateral ligament, is
# named for the side it lies on. Medial and lateral are defined relative to the body's
# midline, so which side of the *image* they fall on depends on which knee was scanned.
# Unless that is normalised, those five labels learn from an axis the model cannot observe.
# 
# The correction differs by plane. Coronally and axially the medial-lateral direction lies in
# the image plane, so flipping the last axis maps one knee onto the other. Sagittally it is
# the *slice* axis: each slice is unchanged by mirroring, and what differs is the order in
# which the stack traverses the joint, so the slice order is reversed instead.
# 
# `Laterality` is a Type 2C attribute: it may legitimately be absent, and here it is absent
# on half the studies — by whole vendors rather than scattered series. Leaving those alone
# silently declares them left-sided, so every right knee among them enters the model
# mirrored. The patient coordinate system supplies the missing tag: position and orientation
# are recorded per image and $+x$ points to the patient's left, so the sign of the image
# centre's $x$ says which knee this is.
# 
# $$c \;=\; \mathbf{p} \;+\; \mathbf{r}\,\Delta_c \frac{N_c}{2} \;+\; \mathbf{d}\,\Delta_r \frac{N_r}{2},
# \qquad \text{side} = \begin{cases} \text{right} & c_x < 0\\ \text{left} & c_x > 0\end{cases}$$
# 
# with $\mathbf{p}$ the image position, $\mathbf{r}$ and $\mathbf{d}$ the row and column
# direction cosines, and $\Delta$ the pixel spacing. The centre is used rather than
# $\mathbf{p}$ itself, which is a corner half a field of view away — enough to change the sign
# on a knee scanned near the midline. The median over a study's series is thresholded, and a
# study centred within a short distance of the midline is left unresolved rather than guessed,
# because inside that band the sign is no better than chance.
# 


# %% cell 21
def normalise_laterality(img, plane, lat):
    """Map every knee onto a left-knee convention.

    Coronal and axial views mirror under a horizontal flip. Sagittal stacks are not
    mirror images of each other - the slice order runs medial-to-lateral in opposite
    directions - so the channel order is reversed instead.
    """
    if lat != "R":
        return img
    if plane in ("Coronal", "Axial"):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])



# %% [markdown]
# ## 5b. Reading once, training many times
# 
# The cost of this pipeline is dominated by reading, not arithmetic. A study holds several
# series and each series tens of slices, so a study is on the order of a hundred and fifty
# files. That is affordable once; it is not affordable once per epoch, and fine-tuning needs
# the same pixels every epoch. So the slot images are decoded a single time into memory and
# held as `uint8`:
# 
# $$\text{bytes} \;=\; N_{\text{study}} \times N_{\text{slot}} \times S \times P^{2}$$
# 
# with $S$ slices kept per slot at $P$ pixels. The exponent on $P$ is what makes this a real
# constraint rather than a detail — the cache grows with the *square* of resolution and only
# linearly with slices, so coverage is the cheap axis and resolution the expensive one.
# 
# The cached slices form one three-channel encoder input per slot, and the layout generalises
# to several such groups per slot: training draws one per step, which doubles as augmentation
# along the stack, and inference averages over them. What fixes the number of groups is a
# budget rather than a capacity — the cache is allowed a fraction of the memory reported free,
# deliberately below the whole of it, since it is the one allocation large enough that
# overshooting ends the run rather than slowing it, and the encoder, its activations and the
# buffers in flight come from the same pool. The size compared against that budget is the sum
# of *both* caches, training and test, because both are resident at once.
# 
# A valid submission file is written before any of this begins and overwritten once real
# predictions exist, so the run always leaves a scoreable file behind.
# 


# %% cell 23
# Where the geometric slice order may be remembered between runs. Unset on the platform,
# because each run gets a fresh machine and there is nothing to remember; set off it,
# where the same corpus is cached again at every resolution and slice count and the order
# is a function of neither. It is opt-in so that the scored run's behaviour is decided by
# the code rather than by whether a file happens to be lying about.
ORDER_CACHE = os.environ.get("RSNA_ORDER_CACHE") or None


def build_cache(slot_map, plane_map, lat_map, tag):
    """Decode every (study, slot) once into an in-memory uint8 array.

    Fine-tuning revisits the same pixels every epoch. Reading them from the mount each
    time would make the epoch count a function of I/O rather than of learning, so they
    are decoded once and held as bytes: intensity has already been normalised into
    [0, 1], and eight bits cost nothing a bilinear resize has not already cost.

    CACHE_SLICES positions are kept per slot, which the training loop reads as N_GROUP
    groups of GROUP consecutive channels.
    """
    studies = sorted(slot_map)
    sidx = {s: i for i, s in enumerate(studies)}
    cache = np.zeros((len(studies), N_SLOT, CACHE_SLICES, IMG, IMG), np.uint8)
    mask = np.zeros((len(studies), N_SLOT), np.float32)
    log(f"{tag}: cache {cache.shape} = {cache.nbytes / 1024 ** 3:.1f} GB")

    jobs = [(st, k, plane, slot_map[st][name])
            for st in studies
            for k, (name, plane, _, _) in enumerate(SLOTS)
            if name in slot_map[st]]
    n_job = len(jobs)

    # Ordering first, and as its own pass. It reads one header per slice of every chosen
    # series - far more file opens than the decode that follows - and on a network mount
    # that is latency, not work, so it gets its own wider pool.
    t_ord = time.time()
    n_slice_total = sum(len(j[3]["files"]) for j in jobs)
    log(f"{tag}: ordering {len(jobs)} slot-series ({n_slice_total} slice headers)")
    ok = done = 0
    CHUNK_O = 1024

    # A remembered order, when one is offered. The projection depends on the DICOM
    # geometry alone, so it is the same at every resolution and every slice count, and
    # it costs one header read per slice - the largest single cost in this pass. An entry
    # is validated by the number of files present, so a tree that has changed under it is
    # recomputed rather than trusted: order is derived data, and a stale entry would be
    # invisible in the way that matters most.
    seen = {}
    if ORDER_CACHE and Path(ORDER_CACHE).is_file():
        try:
            import json as _json
            seen = _json.loads(Path(ORDER_CACHE).read_text())
        except (OSError, ValueError):
            seen = {}
        hit = 0
        for _, _, _, rec in jobs:
            e = seen.get(rec["SeriesInstanceUID"])
            if e and len(e["files"]) == len(rec["files"]):
                rec["ordered"] = e["files"]
                ok += int(e["good"])
                hit += 1
        jobs = [j for j in jobs if "ordered" not in j[3]]
        log(f"{tag}: {hit} slot-series ordered from {ORDER_CACHE}, {len(jobs)} to read")

    with ThreadPoolExecutor(max_workers=ORDER_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK_O):
            block = jobs[c0:c0 + CHUNK_O]
            for (_, _, _, rec), (files, good) in zip(
                    block, pool.map(lambda j: order_slices(j[3]), block)):
                rec["ordered"] = files
                ok += int(good)
                done += 1
                if ORDER_CACHE:
                    seen[rec["SeriesInstanceUID"]] = {"files": files, "good": bool(good)}
            # The ceiling is whichever comes first: the pass's own budget, or the share
            # of what is left of the run that it may take. The second is what makes the
            # first safe to set generously - a mount slow enough to matter cannot spend
            # the training time, because the budget shrinks as the run does.
            budget = min(ORDER_BUDGET_S, max(60.0, (TIME_BUDGET - (time.time() - T0)) * 0.35))
            if time.time() - t_ord > budget:
                log(f"{tag}: ordering budget spent at {done}/{len(jobs)}; "
                    f"the rest keep file order")
                break
    if ORDER_CACHE and done:
        import json as _json
        _t = Path(ORDER_CACHE).with_suffix(".tmp")
        _t.write_text(_json.dumps(seen))
        _t.replace(Path(ORDER_CACHE))
    log(f"{tag}: ordered {ok}/{n_job} by geometry "
        f"({n_job - ok} kept arbitrary) in {time.time() - t_ord:.0f}s")

    jobs = [(st, k, plane, slot_map[st][name])
            for st in studies
            for k, (name, plane, _, _) in enumerate(SLOTS)
            if name in slot_map[st]]
    log(f"{tag}: decoding {len(jobs)} slot-series")
    n_failed_before = len(DECODE_FAILED)

    CHUNK = 512
    done = 0
    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK):
            block = jobs[c0:c0 + CHUNK]
            for (st, k, plane, _), img in zip(
                    block, pool.map(lambda j: read_slot(j[3], CACHE_SLICES, IMG), block)):
                done += 1
                if img is None:
                    continue
                cache[sidx[st], k] = normalise_laterality(img, plane,
                                                          lat_map.get(st)).numpy()
                mask[sidx[st], k] = 1.0
            if done % 4096 < CHUNK:
                log(f"  {tag} {done}/{len(jobs)}")
            if time.time() - T0 > TIME_BUDGET:
                log(f"  {tag}: time budget reached during decode")
                break
    n_failed = len(DECODE_FAILED) - n_failed_before
    log(f"{tag}: {int(mask.sum())}/{len(jobs)} slots filled"
        + (f"; {n_failed} series had a slice that would not decode" if n_failed else ""))
    gc.collect()
    return studies, cache, mask



# %% [markdown]
# ## 6. Aggregating slots into twelve decisions
# 
# A study arrives as up to six slot embeddings $x_s \in \mathbb{R}^{d}$ with a presence mask
# $m_s \in \{0,1\}$. Pooling them identically would discard the reason the protocol has three
# planes: each finding is read on particular sequences, and a mean over slots dilutes the one
# carrying the evidence with five that do not.
# 
# Project each slot, add a learned slot identity, give every diagnosis $o$ its own query
# $q_o \in \mathbb{R}^{H}$, and let it attend over the slots with absent ones masked out of
# the softmax:
# 
# $$h_s \;=\; \phi(x_s) + e_s, \qquad
# \alpha_{o,s} \;=\; \frac{\exp\!\big(\langle h_s, q_o\rangle / \sqrt{H}\big)\, m_s}
# {\sum_{s'} \exp\!\big(\langle h_{s'}, q_o\rangle / \sqrt{H}\big)\, m_{s'}},$$
# 
# $$c_o \;=\; \sum_s \alpha_{o,s}\, h_s, \qquad
# \ell_o \;=\; \langle c_o, w_o \rangle + b_o .$$
# 
# The masked softmax renormalises over whatever the study actually contains, so a missing
# axial series shifts a diagnosis's attention onto the sequences that are present instead of
# feeding it a zero vector.
# 
# **The head is deliberately this small.** The label is attached to the *study*, so nothing
# in the supervision says which part of a study carries the finding, and a richer aggregation
# would have no signal to learn that from. Where the supervision is coarse, so is the head.
# 


# %% cell 25
class SlotHead(nn.Module):
    """Per-diagnosis attention over the slot embeddings of one study.

    Each finding is read on particular sequences - cruciates sagittally, collateral
    ligaments and the meniscal body coronally, patellar cartilage axially - so pooling
    the slots identically would dilute the one that carries the evidence with the rest.

    The aggregation is deliberately this simple. With a study-level label there is no
    signal telling the model which part of a study matters, so extra attention
    parameters below the slot level would have nothing to learn from and would spend
    their capacity fitting noise.
    """

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        # An imported member carries a fixed per-(diagnosis, slot) tilt on the attention
        # logits, set from the anatomy table below rather than learned. It is a buffer, so
        # it travels in the state dict and must exist for that member to load; exp(0.55)
        # gives a preferred slot about 1.73x the weight of an unpreferred one, which
        # biases the softmax without ever excluding a slot.
        p_ = torch.zeros(n_out, n_slot)
        if prior and n_slot == len(SLOTS) and n_out == len(TARGETS):
            for t, slots in SLOT_PRIOR_TABLE.items():
                if t in TARGETS:
                    p_[TARGETS.index(t), list(slots)] = SLOT_PRIOR_STRENGTH
        self.prior = prior
        if prior:
            self.register_buffer("slot_prior", p_)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias



# %% cell 26
class Model(nn.Module):
    """Encoder plus head, trained end to end.

    A study arrives as a bag of slot images. The bag is flattened for the encoder and
    folded back before the head, so the encoder never sees the study structure and the
    head never sees pixels.
    """

    def __init__(self, backbone, dim, pool="cls_mean", prior=False):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask, img_size=None):
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            # The cache is held at the highest resolution any configuration needs; the
            # rest downsample from it, so every configuration sees the same pixels
            # through a different sampling grid rather than a different crop.
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                              align_corners=False)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        parts = [out[:, 0], patch.mean(1)]
        if self.pool == "cls_mean_focal":
            # The upper tail of each channel over the patch grid, taken per channel
            # rather than by selecting whole patches: a finding occupies a small part of
            # the field, so a plain mean over 256 patches dilutes it by two orders of
            # magnitude, and this keeps the top eighth of each channel's responses.
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        feat = torch.cat(parts, dim=1).reshape(B, S, -1)
        return self.head(feat, mask)



# %% [markdown]
# ### Why the encoder is trained rather than frozen
# 
# A frozen self-supervised encoder is bounded by something no work downstream can reach.
# Resolution, encoder size, slice coverage and slot aggregation change how much the model
# looks and how closely, but none changes the vocabulary it looks *with*, so every one of those
# axes runs into the same ceiling — and that ceiling should be expected to bind here, since the
# encoder learned its features from natural images, where nothing resembles the signal a torn
# meniscus makes on a proton-density sequence.
# 
# So the encoder is adapted, with two restraints. **Only the last blocks move** — early blocks
# of a vision transformer are generic edge and texture filters, late blocks are where
# semantics live, and there may not be enough supervision here to improve the early ones while
# there is certainly enough to damage them. **The encoder learns far more slowly than the
# head** — the head is random at initialisation and has everything to learn, the encoder starts
# from a good solution and needs only to be moved off it, so a single learning rate would
# either leave the head untrained or destroy the encoder in the first few hundred steps.
# 
# Targets remain the report-derived labels of §2, weighted by the per-finding confidence that
# comes with them, so a report that never mentions a finding pulls weakly on that output
# rather than asserting a negative there. The studies carrying per-condition annotations are
# weighted above every derived row, since they are the only labels read from the images.
# 


# %% cell 28
def build_model(unfreeze_last, source=None, variant="small", pool="cls_mean",
                prior=False):
    """Load the encoder and open the last `unfreeze_last` blocks for training.

    The early blocks of a self-supervised transformer are generic edge and texture
    filters; the late blocks carry semantics. Opening only the late ones is the cautious
    choice - there may not be enough supervision here to improve the early ones and there
    is certainly enough to damage them - but how far the line should sit is a question
    the corpus has to answer rather than the intuition.

    `source` names where the weights come from. Left unset it is the attached model
    directory, which is the only thing available here. It is a parameter so that a run
    off the platform builds the same object from the same code rather than from a second
    definition that has to be kept in step by hand.
    """
    from transformers import AutoModel
    p = source if source is not None else find_dinov2(variant)
    if p is None:
        raise FileNotFoundError("DINOv2 weights not attached")
    bb = AutoModel.from_pretrained(str(p))
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    dim = bb.config.hidden_size
    trainable = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    log(f"backbone: {n_layer} blocks, last {unfreeze_last} trainable "
        f"({trainable / 1e6:.1f}M params), feature dim {dim * POOL_PARTS[pool]}")
    return Model(bb, dim, pool=pool, prior=prior)



# %% [markdown]
# ## 6b. Reading weights instead of learning them
# 
# Nothing requires the scored run to be the run that learned the weights. A notebook may
# attach a dataset, weights are a dataset, and the part that genuinely cannot be done in
# advance is the part depending on studies nobody has seen — reading them, and predicting.
# Learning here instead caps the model at what one accelerator fits inside the time limit,
# and spends that time again on every submission for a result that does not change. So when
# a package of trained members is attached this section reads it; when none is, the sections
# below train one.
# 
# **Members, not a model.** Each member carries the preprocessing it was fitted on. Two
# members fitted at different resolutions cannot share a decode; two fitted alike can. So
# members are grouped by the pixels they need, each group is decoded once, and a member added
# later joins the list unchanged.
# 
# What groups them is more than the resolution and the crop. Four further decisions settle
# what a slice *is* — which one comes next along the stack, which knees are mirrored, whether
# a slot may be filled from a sequence that does not match its predicate, and what stands in
# for a slice that will not decode — and none of them changes a single shape. A member fitted
# under one reading and decoded under another therefore loads cleanly, runs, and returns a
# submission computed from the wrong image, so the reading travels with the member.
# 
# **A member must prove it is the model it was.** Loading a state dictionary succeeds
# whenever the shapes line up, and shapes line up across every difference that matters — a
# changed normalisation, resize, or slice band. None of those raise. So each member carries
# the answer it gave to a seeded question, recomputed before use; a mismatch stops the run
# rather than being averaged into a submission.
# 
# **How a member is read.** Training draws one group of consecutive slices per step. At
# inference, looking more costs no extra decoding — the cache already holds $S$ slices, so one
# group from the middle is $1$ forward pass, the disjoint groups are $S/G$, and every
# consecutive run of $G$ slices is $S-G+1$. Where the average is taken is free but not
# neutral: averaging logits then squashing is a geometric mean of odds, averaging
# probabilities an arithmetic mean of risk, and they order studies differently. Members are
# combined by rank, since §1 established order is all the metric reads.
# 


# %% cell 30
FINGERPRINT_TOL = 2e-3


def fingerprint(model, dev, img_size, n_slot=None, group=None, seed=None):
    """The model's output on a fixed synthetic bag, as a portable identity.

    Weights that are loaded but read through the wrong preprocessing produce predictions,
    not errors. The submission is well formed, the log says nothing, and the difference is
    a number no output of the run reveals. Scaling that never happens, or happens twice,
    is enough on its own and changes no shape anywhere.

    So a set of weights carries the answer it gave to a question with no data in it. The
    input is generated from a seed rather than read, so it is the same on any machine, and
    it is pushed through the whole forward path - the byte scaling, the ImageNet
    normalisation, the resize, the encoder, the slot attention. Any of those differing
    moves the output by order one. Numerics differing between two GPUs moves it by about
    1e-5, which is why the tolerance sits between them rather than at zero.

    This checks that the model computes what it computed when it was fitted. It cannot
    check that the pixels reaching it are the right pixels; `read_slot` and the header
    pass answer to their own tests.
    """
    n_slot = N_SLOT if n_slot is None else n_slot
    group = GROUP if group is None else group
    seed = SEED if seed is None else seed
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randint(0, 256, (2, n_slot, group, img_size, img_size),
                         generator=g, dtype=torch.uint8).to(dev)
    mask = torch.ones(2, n_slot, device=dev)
    mask[1, -1] = 0.0                       # exercise the masked branch of the softmax
    was_training = model.training
    model.eval()
    with torch.no_grad():
        # float32 throughout: autocast would make the value depend on which device
        # happened to run it, and the point of the number is that it does not.
        out = model(imgs, mask, img_size).float().cpu().numpy()
    if was_training:
        model.train()
    return out


def check_fingerprint(model, dev, img_size, expected, tol=FINGERPRINT_TOL, tag=""):
    """Compare against a stored fingerprint; raise when the model is not the same map."""
    got = fingerprint(model, dev, img_size)
    exp = np.asarray(expected, np.float32)
    if got.shape != exp.shape:
        raise WeightsError(f"{tag}fingerprint shape {got.shape} != stored {exp.shape}: "
                           f"the architecture is not the one these weights were fitted to")
    d = float(np.abs(got - exp).max())
    if d > tol:
        raise WeightsError(
            f"{tag}fingerprint differs by {d:.4g} (tolerance {tol:g}). The weights load "
            f"but do not compute what they computed when fitted - preprocessing, "
            f"resolution or architecture has moved between the two runs.")
    log(f"{tag}fingerprint matches within {d:.2g}")
    return d


class WeightsError(RuntimeError):
    """Raised when attached weights cannot be trusted to be the ones that were fitted.

    Deliberately fatal for the same reason as LabelSourceError: a run that predicts from
    a mismatched model completes, writes a plausible submission, and differs from a
    correct one only in a number no output of the run reveals.
    """


def find_weights(name="manifest.json"):
    """Locate a mounted weights package, or return None if none is attached.

    Same shape as `find_label_table`: the notebook must keep working for a reader who
    attaches nothing, so absence is a path rather than an error. What must not be silent
    is a package that is attached and unusable, and that is what `load_weights` refuses.
    """
    import json
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name not in files:
            continue
        # The manifest decides, not the filenames beside it. Testing for a naming
        # convention makes the search agree with whatever the packager happened to call
        # its files last, which is a second definition of what a package is.
        try:
            man = json.loads((Path(root) / name).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(man.get("members"), list) and man["members"]:
            missing = [m["file"] for m in man["members"]
                       if not (Path(root) / m["file"]).is_file()]
            if missing:
                raise WeightsError(
                    f"{root} holds a manifest listing {len(man['members'])} members but "
                    f"{len(missing)} of their files are absent (first {missing[0]!r})")
            return Path(root)
    return None


# How a member is read at inference. Overlapping windows over the slices the cache
# already holds cost forward passes and no extra decoding, which is the cheap direction
# to spend; and averaging probabilities rather than logits is an arithmetic mean of risk
# rather than a geometric mean of odds, which orders studies differently. Both were
# chosen by measuring them on the folds each member held out rather than by argument.
TTA_OVERLAP = True
TTA_POOL = "prob"


def window_starts(n_slice, group, overlap=None):
    """Where each TTA window begins."""
    overlap = TTA_OVERLAP if overlap is None else overlap
    if overlap and n_slice >= group:
        return list(range(n_slice - group + 1))
    return [g * group for g in range(max(n_slice // group, 1))]


@torch.no_grad()
def predict_member(model, cache, mask, idx, dev, img_size, group=None, pool=None,
                   starts=None):
    """One member's predictions, averaged over its TTA windows.

    `starts` is a parameter rather than always derived here so that a caller which has
    measured what the remaining time affords can read a member over fewer windows. Passing
    it is how the reduction becomes visible in one place instead of being re-derived.
    """
    group = GROUP if group is None else group
    pool = TTA_POOL if pool is None else pool
    starts = window_starts(cache.shape[2], group) if starts is None else list(starts)
    if not starts:
        raise ValueError("predict_member was given no windows to average over")
    model.eval()
    out = []
    for b in range(0, len(idx), EVAL_BATCH):
        sel = idx[b:b + EVAL_BATCH]
        m = torch.from_numpy(mask[sel]).to(dev)
        acc = None
        for st in starts:
            rows = torch.from_numpy(
                np.ascontiguousarray(cache[sel, :, st:st + group])).to(dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(rows, m, img_size).float()
            v = z if pool == "logit" else torch.sigmoid(z)
            acc = v if acc is None else acc + v
        v = acc / len(starts)
        out.append((torch.sigmoid(v) if pool == "logit" else v).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)


def infer_from_package(path, dev):
    """Predict the test split from an attached package of trained members.

    The alternative below - learning the weights inside the scored run - spends the whole
    allowance on the training corpus every time the notebook is submitted, and caps the
    model at what nine hours on one accelerator can fit. Neither is necessary: a notebook
    may attach a dataset, and weights are a dataset. What the scored run then does is the
    part that cannot be done in advance, because the studies are not known in advance.

    Members are grouped by the pixels they need. Two members fitted at different
    resolutions are different functions of the same study and cannot share a decode; two
    fitted alike can, and that is the whole reason the grouping exists rather than a
    decode per member.
    """
    import json
    man = json.loads((Path(path) / "manifest.json").read_text())
    members = man["members"]
    log(f"weights package: {len(members)} member(s) from {path}")

    test_df = pd.read_csv(ROOT / "test.csv")
    test_series = pd.read_csv(ROOT / "test_series.csv")
    plane_map = dict(zip(test_series["SeriesInstanceUID"],
                         test_series["Anatomical_Plane"]))
    hte = annotate(walk("test_series"))
    log(f"test header pass: {len(hte)} series")

    groups = {}
    for m in members:
        groups.setdefault(m["pixel_group"], []).append(m)

    per_member = []
    fixed_s = per_win_s = None
    for gi, (key, gm) in enumerate(groups.items(), 1):
        cfg = json.loads(key)
        adopt_config_globals(cfg)
        log(f"decode group {gi}/{len(groups)}: {cfg['img']}px x {cfg['slices']} slices, "
            f"crop {cfg['crop_mm']} mm -> {len(gm)} member(s)")
        st_te, Cte, Mte = build_cache(pick_slots(hte, plane_map), plane_map,
                                      lat_of(hte, "test "), f"test g{gi}")
        idx = np.arange(len(st_te))

        # What the remaining time affords. A member costs a fixed part - loading the
        # checkpoint, building the encoder, checking the fingerprint - plus a part that
        # scales with the number of TTA windows it is read over. Only the second is worth
        # trading, and only the first member can measure either, because the size of the
        # test set is not something the notebook is told.
        #
        # Both parts are timed separately. Dividing a member's whole wall time by its
        # window count prices the fixed part as if it scaled, so every reduction inflates
        # the estimate that caused it and the estimate runs away from the truth.
        #
        # Windows are surrendered before members. A dropped window costs one view of a
        # study that other windows also see; a dropped member costs an independent vote,
        # which is what an ensemble is made of. The question asked before each member is
        # whether ONE more fits - not whether all the rest do, which would abandon an
        # ensemble that could still have run most of itself.
        starts = window_starts(Cte.shape[2], GROUP)
        order = sorted(gm, key=lambda m: -(m.get("holdout") or 0))
        left_after = sum(len(g) for j, (_, g) in enumerate(groups.items(), 1) if j > gi)
        for k, m in enumerate(order):
            left = TIME_BUDGET - (time.time() - T0)
            remaining = (len(order) - k) + left_after
            if fixed_s is not None and per_win_s is not None:
                # Leave a tenth of what is left unspent: the estimate comes from one
                # member on one machine, and a submission written late is not written.
                afford = max(left * 0.9, 0.0)
                need = fixed_s + len(starts) * per_win_s
                if need * remaining > afford:
                    room = afford / max(remaining, 1)
                    n_win = int((room - fixed_s) / per_win_s) if per_win_s > 0 else 0
                    n_win = max(1, min(len(starts), n_win))
                    if fixed_s + per_win_s > afford:
                        log(f"  {left / 60:.0f} min left: stopping after {k} of "
                            f"{len(order)} in this decode group; not one more member "
                            f"fits, at any window count")
                        break
                    if n_win < len(starts):
                        mid = (len(starts) - n_win) // 2
                        log(f"  {left / 60:.0f} min left, {remaining} member(s) to go: "
                            f"{n_win} window(s) each instead of {len(starts)}")
                        starts = starts[mid:mid + n_win]
            t0 = time.time()
            ck = torch.load(Path(path) / m["file"], map_location="cpu",
                            weights_only=False)
            model = build_model(int(m["config"]["unfreeze_last"]),
                                variant=m["config"]["variant"],
                                pool=m["config"].get("pool", "cls_mean"),
                                prior=bool(m["config"].get("prior", False))).to(dev)
            model.load_state_dict(ck["model"])
            check_fingerprint(model, dev, IMG, ck["fingerprint"], tag=f"{m['id']}: ")
            t_ready = time.time()
            p = predict_member(model, Cte, Mte, idx, dev, IMG, starts=starts)
            per_member.append({"id": m["id"], "ids": st_te, "pred": p,
                               "holdout": m.get("holdout")})
            fixed_s = t_ready - t0
            per_win_s = (time.time() - t_ready) / max(len(starts), 1)
            log(f"  {m['id']} fold {m['fold']}: predicted {len(idx)} studies over "
                f"{len(starts)} window(s) in {time.time() - t0:.0f}s")
            del model, ck
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        del Cte, Mte
        gc.collect()

    # Rank rather than probability, because the metric reads order and two members
    # calibrated differently would otherwise have unequal say. Every member covers every
    # test study, so the mean is over the same set each time and needs no weighting to be
    # comparable; weighting by a fold's holdout would import into the test set a
    # difference measured on a few hundred training studies.
    all_ids = sorted({s for m in per_member for s in m["ids"]})
    pos = {s: i for i, s in enumerate(all_ids)}
    acc = np.zeros((len(all_ids), len(TARGETS)), np.float64)
    for m in per_member:
        r = pd.DataFrame(m["pred"]).rank(pct=True).to_numpy()
        acc[[pos[s] for s in m["ids"]]] += r
    acc /= max(len(per_member), 1)

    sub = write_submission(acc, all_ids, test_df, "submission.csv")
    log(f"submission.csv = rank mean of {len(per_member)} member(s); {sub.shape}; "
        f"nulls {int(sub[TARGETS].isna().sum().sum())}")
    return sub


def adopt_config_globals(cfg):
    """Point the pixel path at what one group of members was fitted on."""
    global IMG, CACHE_IMG, GROUP, CACHE_SLICES, N_GROUP, CROP_MM, SLICE_BAND, RULES
    CACHE_IMG = IMG = int(cfg["img"])
    GROUP = int(cfg["group"])
    CACHE_SLICES = int(cfg["slices"])
    N_GROUP = max(CACHE_SLICES // GROUP, 1)
    CROP_MM = float(cfg["crop_mm"])
    SLICE_BAND = tuple(float(x) for x in cfg["band"])
    # The four decisions that change what a slice is. A member fitted under one reading
    # and decoded under another gets pixels its weights never saw, with every shape
    # still agreeing, so an unrecognised name is refused rather than defaulted.
    rules = cfg.get("rules") or RULES_NATIVE
    unknown = {k: v for k, v in rules.items()
               if k not in RULES_NATIVE
               or v not in (RULES_NATIVE[k], RULES_LEGACY[k])}
    if unknown:
        raise WeightsError(f"the members record pixel rules this pipeline cannot "
                           f"reproduce: {unknown}")
    RULES = {**RULES_NATIVE, **rules}
    if [s[0] for s in SLOTS] != list(cfg["slots"]):
        raise WeightsError(
            f"the members were fitted on slots {cfg['slots']} and this pipeline defines "
            f"{[s[0] for s in SLOTS]}; a weight would be read against the wrong slot")



# %% cell 31
def take_group(cache_rows, g):
    """Slice GROUP consecutive channels out of the cached slices."""
    return cache_rows[:, :, g * GROUP:(g + 1) * GROUP]


def augment(imgs):
    """A small rigid jitter and an intensity scale, applied to a whole bag at once.

    Neither flip is available here, and for different reasons. A horizontal flip would
    reintroduce the nuisance axis that the laterality normalisation removed - it would
    undo, once per batch, what the header pass was run to establish.

    A vertical flip is not a nuisance axis at all. A knee is acquired in a canonical
    orientation, and no study in this corpus looks like its own vertical mirror. An
    augmentation is meant to cover directions along which the label does not change; this
    one moves the input off the distribution the encoder will be asked about, which is a
    different thing. Where a finding sits in the frame is also information rather than
    noise - a Baker cyst is identified by lying in the popliteal fossa, not by its
    appearance alone.

    What is left is jitter that no label depends on: a few degrees of rotation, a few
    per cent of scale and translation. That still prevents memorising the exact framing,
    which is what an augmentation is for, while leaving the anatomy where it was.
    """
    # A bag arrives as [study, slot, GROUP, IMG, IMG]: five axes, not four. The warp is
    # a 2-D operation, so the two leading axes are folded together and restored after -
    # every slot image is an independent acquisition and gets its own jitter.
    lead = imgs.shape[:-3]
    x = imgs.reshape(-1, *imgs.shape[-3:]).float()
    n, dev = x.shape[0], x.device

    rot = (torch.rand(n, device=dev) - 0.5) * 2 * (AUG_ROT_DEG * np.pi / 180)
    # Zoom in only. `border` padding repeats the edge row outward, and the edge of this
    # crop is where the popliteal fossa sits; zooming out would fabricate tissue exactly
    # where a Baker cyst is looked for.
    sc = 1.0 + torch.rand(n, device=dev) * AUG_SCALE
    tx = (torch.rand(n, device=dev) - 0.5) * 2 * AUG_SHIFT
    ty = (torch.rand(n, device=dev) - 0.5) * 2 * AUG_SHIFT
    cos, sin = torch.cos(rot) / sc, torch.sin(rot) / sc
    theta = torch.zeros(n, 2, 3, device=dev, dtype=torch.float32)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    scale = 1.0 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 2 * AUG_INTENSITY
    x = (x * scale).clamp(0, 255)
    return x.reshape(*lead, *x.shape[-3:]).to(imgs.dtype)


@torch.no_grad()
def predict(model, cache, mask, idx, dev, img_size=None):
    """Average the logits over the groups of each slot.

    Training sees one group at a time, which acts as augmentation along the stack;
    inference averages over all of them, so the prediction does not depend on which
    group a single draw happened to pick. Where the cache holds one group per slot the
    two coincide.
    """
    model.eval()
    out = []
    for b in range(0, len(idx), EVAL_BATCH):
        sel = idx[b:b + EVAL_BATCH]
        m = torch.from_numpy(mask[sel]).to(dev)
        acc = None
        for g in range(N_GROUP):
            # Gathered a group at a time rather than whole and then sliced. The two are
            # the same pixels, but taking the whole of a study out of the cache allocates
            # every slice it holds - most of which this pass will not look at until a
            # later iteration, by which time they have been fetched again. Measured over
            # a cache of twelve slices, the difference between the two is the difference
            # between the step being bound by memory and being bound by the encoder.
            rows = torch.from_numpy(np.ascontiguousarray(
                cache[sel, :, g * GROUP:(g + 1) * GROUP])).to(dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(rows, m, img_size).float()
            acc = z if acc is None else acc + z
        out.append(torch.sigmoid(acc / N_GROUP).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)


def macro_auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(np.nanmean([roc_auc_score(y[:, j], p[:, j])
                             if len(set(y[:, j])) > 1 else np.nan
                             for j in range(y.shape[1])]))



# %% [markdown]
# ## 7. Validating without fooling yourself
# 
# Two leaks are specific to this setup, and both inflate a validation number without
# improving anything.
# 
# **Shared reports.** Some reports are byte-identical across studies — a template read for an
# unremarkable knee — so every study in such a group receives the same derived target vector,
# and splitting the group across the divide scores the model on a target whose source it was
# trained on. Studies are therefore assigned by a hash of the report text, which keeps every
# duplicate group whole. One fifth is held out, and the split is fixed rather than rotated.
# 
# **Two references, two meanings.** The **holdout** covers a fifth of the corpus, measures
# agreement with the derived targets, and has enough studies per label to separate a real
# difference from noise — it selects both the epoch within a run and the recipe between runs.
# The **annotation check** measures agreement with a radiologist's reading of the images,
# which is what the competition scores, but only the annotated studies falling in the holdout
# can be used and there are very few; by the standard-error argument of §2 it is reported and
# never allowed to arbitrate.
# 
# The annotated studies stay in training, at elevated weight, because they are the only labels
# read from the images rather than from text. That is exactly why the annotation check must be
# restricted to the holdout: scoring a model on training examples whose answers it saw,
# weighted more heavily than anything else, measures memorisation and reports it as skill.
# 


# %% cell 33
def write_submission(pred, studies, test_df, path):
    """Write one submission file from a prediction matrix.

    Predictions are converted to per-column ranks first: the metric reads only order, so
    ranks discard nothing, and they make files from different configurations directly
    comparable and safe to average.
    """
    sub = pd.DataFrame(pd.DataFrame(pred).rank(pct=True).values, columns=TARGETS)
    sub.insert(0, "StudyInstanceUID", studies)
    sub = test_df[["StudyInstanceUID"]].merge(sub, on="StudyInstanceUID", how="left")
    sub[TARGETS] = sub[TARGETS].fillna(0.5)
    sub.to_csv(path, index=False)
    return sub


def write_benchmark_submission():
    """Write the 0.5 benchmark file immediately.

    A submission that never writes scores nothing at all, which is strictly worse than
    scoring badly. The try/except around main() covers exceptions, but a kill for memory
    is a SIGKILL and never reaches it. So a valid file exists from the first second and
    is overwritten only once real predictions are ready.
    """
    t = pd.read_csv(ROOT / "test.csv")
    for c in TARGETS:
        t[c] = 0.5
    t.to_csv("submission.csv", index=False)


def main():
    write_benchmark_submission()

    # Weights, if any were attached; otherwise the run learns its own below. Both paths
    # are kept because the second is what makes this notebook readable on its own - a
    # fork with nothing attached still trains and still scores - and because the first
    # cannot be checked by anyone who does not have the package.
    pkg = find_weights()
    if pkg is not None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        infer_from_package(pkg, dev)
        log("done")
        return

    # Settle where the labels come from before anything expensive runs. The check costs
    # one CSV header read; discovering the same problem after the cache is built would
    # cost the whole decode pass, and discovering it never would cost the run.
    read_labels(pd.read_csv(ROOT / "train.csv", usecols=["StudyInstanceUID", "Report"]))

    test_df = pd.read_csv(ROOT / "test.csv")
    test_series = pd.read_csv(ROOT / "test_series.csv")
    train_df = pd.read_csv(ROOT / "train.csv")
    train_series = pd.read_csv(ROOT / "train_series.csv")
    log(f"train {train_df.shape} test {test_df.shape}")

    both = pd.concat([train_series, test_series])
    plane_map = dict(zip(both["SeriesInstanceUID"], both["Anatomical_Plane"]))

    log("header pass: test")
    hte = annotate(walk("test_series"))
    log(f"  {len(hte)} test series")
    log("header pass: train")
    htr = annotate(walk("train_series"))
    log(f"  {len(htr)} train series")

    slots_te, slots_tr = pick_slots(hte, plane_map), pick_slots(htr, plane_map)
    cov = pd.Series([len(v) for v in slots_tr.values()]).describe()
    log(f"train slots per study: mean {cov['mean']:.2f} min {cov['min']:.0f} "
        f"max {cov['max']:.0f}")

    st_tr, Ctr, Mtr = build_cache(slots_tr, plane_map, lat_of(htr, "train "), "train")
    st_te, Cte, Mte = build_cache(slots_te, plane_map, lat_of(hte, "test "), "test")

    # ---- targets ---------------------------------------------------------- #
    t_lab = time.time()
    lab = read_labels(train_df)
    log(f"derived labels for {len(lab)} studies in {time.time() - t_lab:.1f}s")

    gold = train_df.set_index("StudyInstanceUID")[TARGETS]
    gold = gold[gold.notna().all(axis=1)]

    Y = np.zeros((len(st_tr), len(TARGETS)), np.float32)
    W = np.zeros_like(Y)
    for i, st in enumerate(st_tr):
        if st in gold.index:
            Y[i], W[i] = gold.loc[st].values, 3.0
        elif st in lab.index:
            r = lab.loc[st]
            Y[i] = r[TARGETS].values
            W[i] = 0.25 + 0.75 * r[[t + "__conf" for t in TARGETS]].values
    keep = np.where(W.sum(1) > 0)[0]
    log(f"supervised {len(keep)} of {len(st_tr)} studies (annotated {len(gold)})")

    # Grouped on report text: some reports are byte-identical across studies and yield
    # one target vector for all of them, so splitting such a group scores the model on a
    # target whose source it has already trained on.
    import hashlib
    rep = train_df.set_index("StudyInstanceUID")["Report"].fillna("")
    grp = np.array([int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % 5
                    for s in st_tr])
    va = np.array([i for i in keep if grp[i] == 0])
    tr = np.array([i for i in keep if grp[i] != 0])
    if len(va) == 0 or len(tr) < BATCH_STUDIES:
        cut = max(1, len(keep) // 5)
        va, tr = keep[:cut], keep[cut:]
    log(f"train {len(tr)} / holdout {len(va)} studies")

    # The annotated studies stay in training - they are the highest-quality labels in
    # the corpus and there are too few to discard - so the honest annotation check uses
    # only the ones that fell in the holdout. Evaluating on the rest would be scoring the
    # model against examples it was trained on, at triple weight, with the true answer.
    gpos = {s: i for i, s in enumerate(st_tr)}
    va_set = set(va.tolist())
    gi = np.array([gpos[s] for s in gold.index if s in gpos and gpos[s] in va_set])
    gold_y = gold.loc[[st_tr[i] for i in gi]].values.astype(int) if len(gi) else None
    yv = (Y[va] > 0.5).astype(int)
    log(f"annotation check: {len(gi)} of {len(gold)} annotated studies are in the holdout")

    # ---- fine-tune -------------------------------------------------------- #
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, test_preds = {}, {}

    for cfg in RUNS:
        pitch = CROP_MM / cfg["img"]
        log(f"=== {cfg['name']}: {cfg['img']} px, {pitch:.3f} mm/pixel, "
            f"{pitch * 14:.2f} mm per patch token ===")
        torch.manual_seed(SEED)
        model = build_model(UNFREEZE_LAST).to(dev)
        opt = torch.optim.AdamW([
            {"params": [p for p in model.backbone.parameters() if p.requires_grad],
             "lr": LR_BACKBONE},
            {"params": model.head.parameters(), "lr": LR_HEAD},
        ], weight_decay=WEIGHT_DECAY)
        steps = max(EPOCHS * (len(tr) // BATCH_STUDIES), 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[LR_BACKBONE, LR_HEAD], total_steps=steps, pct_start=0.15)
        scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

        best, best_state, best_annot = -1.0, None, float("nan")
        for ep in range(EPOCHS):
            model.train()
            perm = np.random.permutation(tr)
            tot, nstep = 0.0, 0
            for b in range(0, len(perm) - BATCH_STUDIES + 1, BATCH_STUDIES):
                sel = perm[b:b + BATCH_STUDIES]
                rows = torch.from_numpy(Ctr[sel]).to(dev)
                g = int(torch.randint(N_GROUP, (1,)).item())
                imgs = augment(take_group(rows, g))
                m = torch.from_numpy(Mtr[sel]).to(dev)
                y = torch.from_numpy(Y[sel]).to(dev)
                w = torch.from_numpy(W[sel]).to(dev)
                with torch.autocast("cuda", enabled=dev.type == "cuda"):
                    loss = (F.binary_cross_entropy_with_logits(
                        model(imgs, m, cfg["img"]), y, reduction="none") * w).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item()
                nstep += 1

            pv = predict(model, Ctr, Mtr, va, dev, cfg["img"])
            d = macro_auc(yv, pv)
            g_auc = float("nan")
            if gold_y is not None and len(gi):
                g_auc = macro_auc(gold_y, predict(model, Ctr, Mtr, gi, dev, cfg["img"]))
            log(f"  epoch {ep + 1}/{EPOCHS}  loss {tot / max(nstep, 1):.4f}"
                f"  holdout {d:.4f}  annot(n={len(gi)}) {g_auc:.4f}")

            # Selection reads the holdout alone. The annotation check is reported because
            # it measures something different - agreement with a reading of the images
            # rather than of the reports - but only a handful of annotated studies land
            # in any one holdout, so its sampling error dwarfs the differences between
            # epochs and it cannot arbitrate between them.
            if d > best:
                best, best_annot = d, g_auc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            if time.time() - T0 > TIME_BUDGET:
                log("  time budget reached")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        results[cfg["name"]] = (best, best_annot)
        test_preds[cfg["name"]] = predict(model, Cte, Mte, np.arange(len(st_te)), dev,
                                          cfg["img"])
        log(f"  {cfg['name']}: best holdout {best:.4f} (annot {best_annot:.4f})")
        del model, opt, sched, scaler, best_state
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    log("---- summary ----")
    for n, (d, g_auc) in results.items():
        log(f"  {n:12s} holdout {d:.4f}   annot {g_auc:.4f}")
    pick = max(results, key=lambda k: results[k][0])
    log(f"best on the holdout: {pick} ({results[pick][0]:.4f})")


    # ---- write every candidate -------------------------------------------- #
    # One file per configuration, plus the holdout's choice as `submission.csv`. A run
    # costs a full decode of the corpus whichever configuration wins, so keeping every
    # arm makes a later change of configuration free rather than another full run.
    for name, pred in test_preds.items():
        sub = write_submission(pred, st_te, test_df, f"submission_{name}.csv")
        log(f"  submission_{name}.csv {sub.shape}; "
            f"nulls {int(sub[TARGETS].isna().sum().sum())}")

    ens = np.mean([pd.DataFrame(p).rank(pct=True).values for p in test_preds.values()],
                  axis=0)
    write_submission(ens, st_te, test_df, "submission_rankmean.csv")
    log(f"  submission_rankmean.csv (rank mean of {len(test_preds)})")

    sub = write_submission(test_preds[pick], st_te, test_df, "submission.csv")
    log(f"submission.csv = {pick}; {sub.shape}; "
        f"nulls {int(sub[TARGETS].isna().sum().sum())}")
    print(sub.head().to_string())



# %% cell 34
try:
    main()
except LabelSourceError:
    # Deliberately not absorbed: see LabelSourceError. A run that trained on the
    # wrong labels would finish and write a submission worth submitting by mistake.
    traceback.print_exc()
    raise
except Exception:
    traceback.print_exc()
    # A submission that fails to write scores nothing at all, so fall back to the
    # benchmark file rather than dying.
    t = pd.read_csv(find_root() / "test.csv")
    for c in TARGETS:
        t[c] = 0.5
    t.to_csv("submission.csv", index=False)
    print("wrote fallback submission.csv")
log("done")

