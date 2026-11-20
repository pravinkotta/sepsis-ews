"""Fetch and verify the guideline corpus used for grounded SBAR drafts.

Sources
-------
1. Surviving Sepsis Campaign 2021, full text via Europe PMC (PMC8486643).
   PMC Open Access Subset: unrestricted research re-use with acknowledgement.
2. SSC Hour-1 Bundle PDF (SCCM), short excerpt with attribution.
3. A project-local escalation SOP, authored here and clearly labelled as a
   project artefact — not clinical guidance.

    python scripts/fetch_guidelines.py            # fetch + verify
    python scripts/fetch_guidelines.py --offline  # verify committed corpus

A short/blocked/changed source exits non-zero; nothing is silently replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "guidelines"

SSC_XML = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8486643/fullTextXML"
SSC_LANDING = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8486643/"
SSC_LICENSE = (
    "PMC Open Access Subset: unrestricted research re-use and secondary "
    "analysis with acknowledgement of the original source (Springer Nature, 2021)."
)
HOUR1_PDF = "https://sccm.org/SCCM/media/SCCM/PDFs/Surviving-Sepsis-Campaign-Hour-1-Bundle.pdf"
HOUR1_LICENSE = "© Society of Critical Care Medicine; short excerpt with attribution."

REQUIRED = {
    "ssc-2021.md": (
        20_000,
        ["lactate", "cultures", "antimicrobials", "30 ml/kg", "vasopressors",
         "mean arterial pressure", "septic shock"],
    ),
    "hour-1-bundle.md": (
        100,
        ["lactate", "blood cultures", "antibiotics", "30 ml/kg", "vasopressors"],
    ),
    "local-escalation-policy.md": (
        200,
        ["sbar", "rapid response", "approval", "token"],
    ),
}


class CorpusError(RuntimeError):
    pass


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _headers() -> dict:
    return {"User-Agent": "sepsis-ews/0.1 (portfolio research; contact: local)"}


def fetch_bytes(url: str, *, min_bytes: int = 1000) -> bytes:
    response = requests.get(url, headers=_headers(), timeout=90)
    if response.status_code != 200:
        raise CorpusError(f"{url} returned HTTP {response.status_code}")
    if len(response.content) < min_bytes:
        raise CorpusError(
            f"{url} returned only {len(response.content)} bytes; expected >= {min_bytes}"
        )
    return response.content


def xml_to_markdown(xml: bytes) -> tuple[str, str, str]:
    soup = BeautifulSoup(xml, "xml")
    title = _clean(soup.find("article-title").get_text(" ", strip=True)) if soup.find(
        "article-title"
    ) else "Surviving Sepsis Campaign 2021"
    doi = ""
    for article_id in soup.find_all("article-id"):
        if article_id.get("pub-id-type") == "doi":
            doi = _clean(article_id.get_text())
            break
    body = soup.find("body")
    if body is None:
        raise CorpusError("Europe PMC XML has no <body>")

    for xref in body.find_all("xref"):
        xref.decompose()
    for fig in body.find_all("fig"):
        fig.decompose()

    lines: list[str] = [f"# {title}", ""]
    if doi:
        lines += [f"DOI: {doi}", ""]

    def walk(node, level: int = 2) -> None:
        for child in node.children:
            name = getattr(child, "name", None)
            if name is None:
                continue
            if name == "sec":
                sec_title = child.find("title", recursive=False)
                if sec_title:
                    heading = _clean(sec_title.get_text(" ", strip=True))
                    if heading:
                        lines.extend([f"{'#' * min(level, 6)} {heading}", ""])
                walk(child, min(level + 1, 6))
            elif name == "p":
                text = _clean(child.get_text(" ", strip=True))
                if text:
                    lines.extend([text, ""])
            elif name == "list":
                for item in child.find_all("list-item", recursive=False):
                    text = _clean(item.get_text(" ", strip=True))
                    if text:
                        lines.append(f"- {text}")
                lines.append("")
            elif name in ("title", "table-wrap", "graphic", "supplementary-material"):
                continue
            else:
                walk(child, level)

    walk(body)
    return title, doi, "\n".join(lines).strip() + "\n"


def pdf_to_markdown(pdf: bytes) -> str:
    reader = PdfReader(BytesIO(pdf))
    pages = [page.extract_text() or "" for page in reader.pages]
    raw = "\n".join(pages)

    lines: list[str] = []
    for line in raw.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        # Drop pure page furniture (column artefacts such as "68  96  14").
        if re.fullmatch(r"[\d\s/.,%-]+", line):
            continue
        lines.append(line)

    # The card is laid out in columns; joining lines into flowing prose is far
    # more useful for retrieval than preserving the PDF's line breaks.
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if len(text.split()) < 50:
        raise CorpusError("Hour-1 PDF produced too little text; layout may have changed")
    return text + "\n"


LOCAL_POLICY = """# Project-local escalation policy (authored artefact)

> **Status:** authored for this case study. It is **not** clinical guidance and
> has not been approved by any hospital. It exists so the retrieval corpus has a
> concrete escalation procedure to cite alongside the SSC guideline.

## 1. Trigger

Escalate when the calibrated sepsis risk is at or above the deployed threshold,
or when a bedside clinician is concerned regardless of the score. The model is a
screening aid; clinician concern overrides it.

## 2. SBAR structure

Every escalation draft contains four sections:

- **Situation** — who the patient is, the current hour, the calibrated risk and
  the top model drivers.
- **Background** — objective facts available at that hour: vitals, lactate,
  cultures, antibiotics, fluids, and relevant history.
- **Assessment** — whether the risk and facts are concerning for evolving
  sepsis, with the guideline citation that supports the statement.
- **Recommendation** — what a clinician should consider, citing the local policy
  or the guideline. The draft may recommend *considering* an action; it may not
  issue a treatment order.

## 3. Human approval

The agent may draft and propose. It may not page, order, or write to the EHR.
A rapid-response page requires a human approval token minted after a clinician
confirms the specific alert. The token is bound to one alert, one patient and
one tool, expires in 15 minutes, and is consumed once. The model never sees the
signing secret.

## 4. Escalation tiers

1. **Bedside review** — the assigned nurse or clinician reviews the draft.
2. **Rapid response** — if the clinician confirms, the approved token releases a
   page to the rapid-response team.
3. **ICU transfer** — a separate clinical decision; this system has no tool for
   admission or transfer.

## 5. Documentation and audit

Every tool call, approval, denial, and page is appended to the audit log. Note
text is hashed rather than copied into the log. A rejected draft is kept with
the reason so the prompt and retrieval can be reviewed.

## 6. Hard prohibitions

- No autonomous antibiotic, fluid, vasopressor, or culture order.
- No autonomous ICU admission, discharge, or alert suppression.
- No page without a bound, single-use approval token.
- No claim of confirmed sepsis; the draft says "concerning for" and defers to
  the clinician.
"""


def write_with_header(path: Path, header: dict, body: str) -> None:
    lines = ["<!--"]
    for key, value in header.items():
        lines.append(f"{key}: {value}")
    lines.append("-->")
    lines.append("")
    path.write_text("\n".join(lines) + body, encoding="utf-8")


def fetch_all() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    retrieved = datetime.now(timezone.utc).isoformat()

    title, doi, ssc = xml_to_markdown(fetch_bytes(SSC_XML, min_bytes=100_000))
    write_with_header(
        OUT / "ssc-2021.md",
        {
            "title": title,
            "source": SSC_LANDING,
            "api": SSC_XML,
            "doi": doi,
            "license": SSC_LICENSE,
            "retrieved_at": retrieved,
        },
        ssc,
    )

    hour1 = pdf_to_markdown(fetch_bytes(HOUR1_PDF, min_bytes=20_000))
    write_with_header(
        OUT / "hour-1-bundle.md",
        {
            "title": "Surviving Sepsis Campaign Hour-1 Bundle",
            "source": HOUR1_PDF,
            "license": HOUR1_LICENSE,
            "retrieved_at": retrieved,
            "note": "Short excerpt for non-commercial research with attribution.",
        },
        "# Surviving Sepsis Campaign Hour-1 Bundle\n\n" + hour1,
    )

    write_with_header(
        OUT / "local-escalation-policy.md",
        {
            "title": "Project-local escalation policy",
            "source": "authored in this repository",
            "license": "Project artefact; not clinical guidance.",
            "retrieved_at": retrieved,
        },
        LOCAL_POLICY,
    )
    print(f"wrote corpus -> {OUT}")


def verify() -> dict:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documents": [],
    }
    failures: list[str] = []
    for name, (min_words, keywords) in REQUIRED.items():
        path = OUT / name
        if not path.exists():
            failures.append(f"{name}: missing")
            continue
        text = path.read_text(encoding="utf-8")
        words = len(text.split())
        lowered = text.lower()
        missing = [kw for kw in keywords if kw.lower() not in lowered]
        if words < min_words:
            failures.append(f"{name}: {words} words < {min_words}")
        if missing:
            failures.append(f"{name}: missing keywords {missing}")
        manifest["documents"].append(
            {
                "file": name,
                "words": words,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "min_words": min_words,
                "keywords_ok": not missing,
            }
        )
        print(f"  {name:<32} {words:>7,} words  {'OK' if not missing else 'MISSING ' + str(missing)}")

    if failures:
        raise CorpusError("; ".join(failures))
    (OUT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="verify without fetching")
    args = parser.parse_args(argv)

    try:
        if not args.offline:
            fetch_all()
        manifest = verify()
    except (CorpusError, requests.RequestException) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    total = sum(doc["words"] for doc in manifest["documents"])
    print(f"corpus verified: {len(manifest['documents'])} documents, {total:,} words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
