#!/usr/bin/env python3
"""Resolve every reference by identifier and freeze the metadata.

    python3 literature_review/resolve_references.py

Writes literature_review/references.json. The manuscript builder renders the
reference list AND every in-text citation from that file and nothing else,
so an in-text year cannot disagree with its reference-list entry, and a key
that is cited but not listed, or listed but not cited, fails the build.

Metadata comes from doi.org content negotiation (CSL-JSON), which returns the
record the publisher deposited with Crossref or, for arXiv, with DataCite.
Nothing here is looked up by title. Where a DOI was not already in hand it was
DISCOVERED by a bibliographic query, accepted only on an exact title match
with the right first author, and then resolved by that DOI like every other
entry; `found_by` records which entries came that way.

Works with no DOI carry a hand-written record, and `provenance` says so.

Where a deposited record is wrong, an override is applied here WITH ITS
REASON, so a correction is visible rather than silent. The first record this
script ever fetched needed one: Crossref gives LoHi-WELD's first author the
family name "Biasuz Block", which the paper itself does not use.

THE YEAR RULE. APA 7, which this journal follows, dates a journal article by
the issue it appears in, and dates a chapter by its book. So the year is
`published-print` when the record has one and `issued` otherwise. That moves
several years away from the ones the literature notes carried, all for the
same reason: an article published online first in one year and assigned to
an issue in the next. Springer's own "cite this article" for Bauer et al.
gives the issue year, 2026.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "references.json"
UA = "weld-benchmark-reference-resolver (mailto:vincentldsouza@nitte.edu.in)"

# (key, doi, found_by). found_by is "given" when the DOI was already recorded
# in the project log or the resolved-citations ledger, and "query" when it was
# discovered this pass by an exact-title bibliographic query.
DOI_ENTRIES = [
    # ---- weld inspection, the entries the project log's corrections list
    ("block2024", "10.1109/access.2024.3407019", "given"),
    ("xie2023", "10.1115/1.4056219", "given"),
    ("antony2026", "10.1080/09507116.2025.2563776", "given"),
    ("bauer2026", "10.1007/s10845-025-02569-6", "given"),
    ("lema2026", "10.1007/s10845-025-02672-8", "given"),
    ("provencal2021", "10.1016/j.promfg.2021.07.041", "given"),
    ("yu2024", "10.1088/1361-6501/ad0613", "given"),
    ("parmarpal2026", "10.1007/978-3-032-08243-5_32", "given"),
    ("mendieta2025", "10.3389/frobt.2025.1632417", "given"),
    ("shen2026", "10.1016/j.ndteint.2025.103521", "given"),
    # ---- weld inspection, discovered this pass from the review's list
    ("palma2024", "10.1016/j.heliyon.2024.e30590", "query"),
    ("lopez2025", "10.3390/s25196183", "query"),
    ("parmar2026", "10.1038/s41598-026-44874-x", "query"),
    ("mustajib2026", "10.1080/09507116.2026.2708882", "query"),
    ("liang2025", "10.3390/s25227024", "query"),
    ("sun2025", "10.3390/s25206314", "query"),
    ("xu2024", "10.1371/journal.pone.0313348", "query"),
    ("mohammed2025", "10.1109/access.2025.3574083", "query"),
    ("totino2023", "10.53375/ijecer.2023.320", "query"),
    ("mery2015", "10.1007/s10921-015-0315-7", "query"),
    ("zhao2025swrd", "10.1007/s10921-025-01186-w", "query"),
    # ---- the resolved-citations ledger, first pass
    ("kennerley2025", "10.48550/arXiv.2506.04737", "given"),
    ("hussain2026", "10.1007/s10462-026-11502-z", "given"),
    ("awad2026", "10.1109/access.2026.3685121", "given"),
    ("ma2022a", "10.1109/CVPRW56347.2022.00532", "given"),
    ("singh2025", "10.1007/978-3-031-72784-9_16", "given"),
    ("ma2022b", "10.1007/s10845-022-02051-7", "given"),
    ("zhao2026", "10.48550/arXiv.2603.21824", "query"),
    ("chakraborty2026", "10.48550/arXiv.2601.09497", "given"),
    ("sun2026", "10.48550/arXiv.2607.04675", "given"),
    # ---- the ledger, second pass
    ("silva2026", "10.3390/app16063022", "given"),
    ("krassnig2026", "10.1007/s10845-025-02778-z", "given"),
    ("jensen2025", "10.1007/978-3-031-73411-3_2", "query"),
    ("saenko2010", "10.1007/978-3-642-15561-1_16", "given"),
    ("heckler2026", "10.1007/s11263-026-02743-0", "given"),
    ("barz2020", "10.3390/jimaging6060041", "given"),
    ("laroca2023", "10.1109/ijcnn54540.2023.10191584", "given"),
    ("ramos2025", "10.1109/iccvw69036.2025.00661", "given"),
    ("nogueira2026", "10.1007/s42979-026-05141-y", "given"),
    ("tampu2022", "10.1038/s41597-022-01618-6", "given"),
    ("adimoolam2023", "10.48550/arXiv.2304.02296", "given"),
    ("cassidy2022", "10.1016/j.media.2021.102305", "given"),
    ("northcutt2021", "10.48550/arXiv.2103.14749", "given"),
    ("wang2025", "10.63367/199115992025103605009", "given"),
    ("wang2022", "10.1109/TIM.2021.3128208", "given"),
    ("wang2019", "10.1109/cvpr.2019.00734", "given"),
    ("chu2016", "10.1007/978-3-319-49409-8_34", "given"),
    ("benhafid2026", "10.3390/app16031236", "given"),
    ("huseljic2025", "10.1007/s11263-024-02219-z", "given"),
    ("munir2022", "10.52202/068431-2805", "query"),
    ("kuzucu2025", "10.1007/978-3-031-72664-4_11", "query"),
    ("oksuz2023", "10.1109/CVPR52729.2023.00894", "given"),
    ("kuppers2020", "10.1109/cvprw50498.2020.00171", "given"),
    ("oza2024", "10.1109/TPAMI.2022.3217046", "given"),
    ("zou2026", "10.1088/2631-8695/ae8b7b", "given"),
    # ---- added this pass, each verified against its abstract
    ("babu2026", "10.1109/iv66570.2026.11623865", "query"),
    ("zhou2022", "10.1109/cvpr52688.2022.00742", "query"),
    # ---- methods, cited in Section 3
    ("zhao2024", "10.1109/cvpr52733.2024.01605", "query"),
    ("ren2017", "10.1109/TPAMI.2016.2577031", "given"),
    ("he2016", "10.1109/CVPR.2016.90", "given"),
    ("lin2017", "10.1109/CVPR.2017.106", "given"),
    ("lin2014", "10.1007/978-3-319-10602-1_48", "given"),
    ("friedman1937", "10.1080/01621459.1937.10503522", "given"),
    ("huangfu2018", "10.1007/s12532-017-0130-5", "given"),
    ("naeini2015", "10.1609/aaai.v29i1.9602", "given"),
    ("field2007", "10.1111/j.1467-9868.2007.00593.x", "given"),
    ("paszke2019", "10.48550/arXiv.1912.01703", "given"),
]

# Works with no DOI. Each is written from its landing page, which is named in
# `provenance`, and each was opened on the date the script records.
MANUAL = {
    "demsar2006": {
        "type": "article-journal",
        "author": [{"family": "Demšar", "given": "Janez"}],
        "title": "Statistical comparisons of classifiers over multiple data sets",
        "container-title": "Journal of Machine Learning Research",
        "volume": "7", "page": "1-30", "year": 2006,
        "URL": "https://jmlr.org/papers/v7/demsar06a.html",
        "_provenance": "JMLR assigns no DOI. Landing page given as URL.",
    },
    "jocher2024": {
        "type": "software",
        "author": [{"family": "Jocher", "given": "Glenn"},
                   {"family": "Qiu", "given": "Jing"}],
        "title": "Ultralytics YOLO11",
        "version": "11.0.0", "year": 2024, "publisher": "Ultralytics",
        "URL": "https://github.com/ultralytics/ultralytics",
        "_provenance": "The citation Ultralytics publishes for YOLO11. No DOI. "
                       "The grid ran ultralytics 8.4.138, which Section 3.8 "
                       "states.",
    },
    "wijaya2024": {
        "type": "dataset",
        "author": [{"family": "Wijaya", "given": "Sukma Adhi"}],
        "title": "Welding defect - object detection",
        "version": "2", "year": 2024, "publisher": "Kaggle",
        "URL": "https://www.kaggle.com/datasets/sukmaadhiwijaya/"
               "welding-defect-object-detection",
        "_provenance": "Kaggle dataset page and metadata endpoint, read 18 "
                       "September 2026 (the project log, Kaggle licence). CC0 1.0.",
    },
    "welddefect2025": {
        "type": "dataset",
        "author": [{"literal": "Weld defect Detection"}],
        "title": "Weld surface defect detection",
        "version": "2", "year": 2025, "publisher": "Roboflow Universe",
        "URL": "https://universe.roboflow.com/weld-defect-detection-onlal/"
               "weld-surface-defect-detection-884h9",
        "_provenance": "Roboflow Universe project page, read 24 September "
                       "2026. The author is the workspace display name, "
                       "which is how Roboflow's own citation names it. CC BY "
                       "4.0. Our copy is export version 2.",
    },
    "faza2024": {
        "type": "dataset",
        "author": [{"literal": "Faza"}],
        "title": "Visual welding defect",
        "version": "1", "year": 2024, "publisher": "Roboflow Universe",
        "URL": "https://universe.roboflow.com/faza/visual-welding-defect",
        "_provenance": "Roboflow Universe project page, read 24 September "
                       "2026. CC BY 4.0. Our copy is export version 1.",
    },
    "kaidal2024": {
        "type": "dataset",
        "author": [{"literal": "kaid Al2"}],
        "title": "Welding defect 2class",
        "year": 2024, "publisher": "Roboflow Universe",
        "URL": "https://universe.roboflow.com/kaid-al2/welding-defect-2class",
        "_provenance": "Roboflow Universe project page, read 24 September "
                       "2026. CC BY 4.0. THIS IS THE UPSTREAM of `twoclass`, "
                       "found this pass. It publishes no dataset version, so "
                       "it cannot be exported directly, and the copy this "
                       "study used is a fork in the author's own workspace. "
                       "The data availability statement names both.",
    },
}

# Corrections to deposited records, each with its reason.
OVERRIDES = {
    "block2024": [
        ("author", [{"family": "Block", "given": "Sylvio Biasuz"},
                    {"family": "da Silva", "given": "Ricardo Dutra"},
                    {"family": "Lazzaretti", "given": "André Eugênio"},
                    {"family": "Minetto", "given": "Rodrigo"}],
         "Crossref records the first author's family name as 'Biasuz Block' "
         "and misspells a co-author as 'Andre Eugnio'. The paper's running "
         "head reads 'S. B. Block et al.', so the family name is Block."),
    ],
    "yu2024": [
        ("author", [{"family": "Yu", "given": "Zeyu"},
                    {"family": "Ma", "given": "Qi"},
                    {"family": "Yuan", "given": "Hongqiang"},
                    {"family": "Du", "given": "Guofeng"}],
         "THE DEPOSITED RECORD SWAPS GIVEN AND FAMILY NAME FOR ALL FOUR "
         "AUTHORS: it records family names Zeyu, Qi, Hongqiang and Guofeng. "
         "That swap is where the project log's 'Zeyu et al.' came from. Scopus "
         "gives Zeyu Yu, Qi Ma, Hongqiang Yuan. Cite as Yu et al."),
    ],
    "antony2026": [
        ("author", [{"literal": "Antony Aroul Raj V."},
                    {"literal": "Pathmanaban P."},
                    {"literal": "Madasamy Raja G."},
                    {"literal": "Hariharan C."},
                    {"literal": "Praveen Kumar M."},
                    {"literal": "Kishore T."},
                    {"literal": "Rohan Akash V."}],
         "THE FIRST AUTHOR IS ANTONY AROUL RAJ V., NOT VENKATARAMANAN. "
         "OpenAlex records the name printed on the paper as 'Antony Aroul "
         "Raj V.' and then resolves it to a different person, 'Abhinau K. "
         "Venkataramanan', and Scopus carries the same wrong resolution. "
         "That is where the project log's 'Venkataramanan et al.' came from. The "
         "names are given exactly as printed, because each family name is "
         "an initial and inverting them would produce 'V., A. A. R.'."),
    ],
    "kuppers2020": [
        ("_family0", "Küppers", "The deposit drops the umlaut."),
    ],
    "zhou2022": [
        ("_author2", {"family": "Krähenbühl", "given": "Philipp"},
         "The IEEE deposit drops both umlauts."),
    ],
    "lin2017": [
        ("_author1", {"family": "Dollár", "given": "Piotr"},
         "The IEEE deposit drops the accent that the COCO record keeps."),
    ],
    "sun2026": [
        ("_author6", {"family": "Le Callet", "given": "Patrick"},
         "DataCite splits the family name as 'Callet' with given 'Patrick "
         "Le'. The family name is Le Callet."),
    ],
    "provencal2021": [
        ("_family0", "Provençal", "The deposit drops the cedilla, which "
         "Scopus and the paper keep. The manuscript draft had also written "
         "'Provencal and Laperriere', losing both accents."),
    ],
    "naeini2015": [
        ("_author0", {"family": "Naeini", "given": "Mahdi Pakdaman"},
         "The deposit gives the compound family name 'Pakdaman Naeini'. The "
         "calibration literature cites this paper as Naeini et al., which "
         "is the form a reader will search for."),
    ],
    "northcutt2021": [
        ("type", "paper-conference", "Published in the NeurIPS Datasets and "
         "Benchmarks track, which assigns no DOI. The arXiv DOI is kept as "
         "the locator, as the 20 September ledger instructs."),
        ("container-title", "Proceedings of the Neural Information Processing "
         "Systems Track on Datasets and Benchmarks", "As above."),
        ("volume", "1", "As above."),
    ],
    "paszke2019": [
        ("type", "paper-conference", "Published in NeurIPS 2019, which "
         "assigns no DOI. The arXiv DOI is kept as the locator."),
        ("container-title", "Advances in Neural Information Processing "
         "Systems", "As above."),
        ("volume", "32", "As above."),
    ],
}

# A chapter's own record names only the series. The book is resolved by ITS
# identifier, the chapter DOI without the chapter suffix, so the book title
# and editors come from the publisher as well.



def _fold_name(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c)).casefold().replace(" ", "")


def tidy(value):
    """Deposits carry HTML entities ('NDT &amp; E International') and
    non-breaking spaces ('BugNIST a\\xa0Large'). Normalise both at source."""
    if isinstance(value, str):
        return " ".join(html.unescape(value).replace(" ", " ").split())
    if isinstance(value, list):
        return [tidy(v) for v in value]
    if isinstance(value, dict):
        return {k: tidy(v) for k, v in value.items()}
    return value


def fetch_csl(doi: str) -> dict:
    req = urllib.request.Request(
        f"https://doi.org/{doi}",
        headers={"Accept": "application/vnd.citationstyles.csl+json",
                 "User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 3:
                raise
            time.sleep(2 + 3 * attempt)
    raise RuntimeError("unreachable")


def date_year(csl: dict, field: str):
    parts = (csl.get(field) or {}).get("date-parts") or [[None]]
    return parts[0][0] if parts and parts[0] else None


def apa_year(csl: dict) -> tuple[int, str]:
    """The issue or book year when there is one, otherwise the issued year."""
    y = date_year(csl, "published-print")
    if y:
        return int(y), "published-print"
    y = date_year(csl, "issued")
    if y:
        return int(y), "issued"
    raise ValueError(f"no date on {csl.get('DOI')}")


KEEP = ("type", "title", "container-title", "volume", "issue", "page",
        "article-number", "publisher", "DOI", "URL", "author", "editor",
        "collection-title", "event")


def main() -> int:
    today = dt.date.today().isoformat()
    out = []
    for key, doi, found_by in DOI_ENTRIES:
        csl = tidy(fetch_csl(doi))
        rec = {k: csl[k] for k in KEEP if k in csl}
        rec["author"] = [{k: a[k] for k in ("family", "given", "literal")
                          if k in a} for a in csl.get("author", [])]
        year, rule = apa_year(csl)
        if rec.get("type") == "book-chapter":
            book_doi = doi.rsplit("_", 1)[0]
            book = tidy(fetch_csl(book_doi))
            rec["book-title"] = book.get("title")
            rec["editor"] = [{k: e[k] for k in ("family", "given") if k in e}
                             for e in book.get("editor", [])]
            rec["series"] = rec.pop("container-title", None)
            rec["book-doi"] = book_doi
        applied = []
        for field, value, reason in OVERRIDES.get(key, []):
            if field == "_family0":
                rec["author"][0]["family"] = value
            elif field.startswith("_author"):
                i = int(field[len("_author"):])
                # Guard: the override must land on the author it names.
                was = rec["author"][i]
                assert _fold_name(was.get("family", "")) in _fold_name(
                    value["family"]) or _fold_name(value["family"]) in \
                    _fold_name(was.get("family", "") + was.get("given", "")), \
                    (key, i, was, value)
                rec["author"][i] = value
            else:
                rec[field] = value
            applied.append({"field": field.lstrip("_"), "reason": reason})
        out.append({
            "key": key, "year": year, "year_rule": rule, "csl": rec,
            "provenance": {"method": "doi.org content negotiation, CSL-JSON",
                           "doi": doi, "found_by": found_by,
                           "resolved": today,
                           "online_year": date_year(csl, "published-online"),
                           "overrides": applied},
        })
        a0 = rec["author"][0] if rec["author"] else {}
        print(f"{key:20s} {year} ({rule:15s}) "
              f"{a0.get('family') or a0.get('literal')}, "
              f"{len(rec['author'])} au | "
              f"{(rec.get('container-title') or rec.get('book-title') or '')[:38]:38s} | "
              f"{rec.get('volume')}({rec.get('issue')}) "
              f"{rec.get('page') or rec.get('article-number')}")
        time.sleep(0.3)
    for key, rec in MANUAL.items():
        rec = dict(rec)
        prov = rec.pop("_provenance")
        year = rec.pop("year")
        out.append({"key": key, "year": year, "year_rule": "stated",
                    "csl": rec,
                    "provenance": {"method": "hand-written, no DOI",
                                   "note": prov, "resolved": today}})
        print(f"{key:20s} {year} (manual)          {rec['title'][:50]}")
    keys = [e["key"] for e in out]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        raise SystemExit(f"duplicate keys: {sorted(dupes)}")
    OUT.write_text(json.dumps({"resolved": today, "entries": out},
                              indent=1, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT.relative_to(HERE.parent)}: {len(out)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
