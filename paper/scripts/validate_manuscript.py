#!/usr/bin/env python3
"""Validate manuscript citations, artifacts, and reproducibility bindings."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manuscript = (PAPER / "main.tex").read_text(encoding="utf-8")
    bibliography = (PAPER / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bibliography))
    cited = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", manuscript):
        cited.update(value.strip() for value in group.split(","))
    missing_citations = sorted(cited - bib_keys)
    required = [
        PAPER / "figures" / "figure1_protocol_maps.pdf",
        PAPER / "figures" / "figure1_protocol_maps.png",
        PAPER / "figures" / "figure2a_performance_ped.pdf",
        PAPER / "figures" / "figure2a_performance_ped.png",
        PAPER / "figures" / "figure2b_paired_ped.pdf",
        PAPER / "figures" / "figure2b_paired_ped.png",
        PAPER / "tables" / "table1_main.tex",
        PAPER / "statistical_report.json",
        PAPER / "main.pdf",
    ]
    missing_artifacts = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    summary = ROOT / "results" / "xai" / "high_gamma_replacement_ped_v1" / "summary.json"
    expected_summary_sha = "2005ef2c8b62207fed1215f896ce855a537ede128f39c12c2d1fe00b3027cc43"
    summary_digest_matches = summary.is_file() and sha256(summary) == expected_summary_sha
    tex_engines = [name for name in ("latexmk", "pdflatex", "xelatex", "lualatex", "tectonic") if shutil.which(name)]
    pdf_pages = None
    references_page_only = False
    if (PAPER / "main.pdf").is_file() and shutil.which("pdfinfo"):
        info = subprocess.run(
            ["pdfinfo", str(PAPER / "main.pdf")],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
        pdf_pages = int(match.group(1)) if match else None
    if (PAPER / "main.pdf").is_file() and shutil.which("pdftotext") and pdf_pages:
        text = subprocess.run(
            [
                "pdftotext",
                "-f",
                str(pdf_pages),
                "-l",
                str(pdf_pages),
                str(PAPER / "main.pdf"),
                "-",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        references_page_only = text.startswith("References") and "Fig." not in text
    placeholders = sorted(
        set(re.findall(r"[^\n]*(?:replace before submission|placeholder)[^\n]*", manuscript, flags=re.I))
    )
    receipt = {
        "status": (
            "draft_ready_author_metadata_pending"
            if not missing_citations
            and not missing_artifacts
            and summary_digest_matches
            and pdf_pages == 4
            and references_page_only
            else "needs_revision"
        ),
        "missing_citations": missing_citations,
        "unused_bibliography_keys": sorted(bib_keys - cited),
        "missing_artifacts": missing_artifacts,
        "summary_digest_matches": summary_digest_matches,
        "tex_engines": tex_engines,
        "pdf_pages": pdf_pages,
        "references_page_only": references_page_only,
        "author_placeholders": placeholders,
        "manuscript_word_count_approx": len(re.findall(r"\b[A-Za-z][A-Za-z-]*\b", manuscript)),
        "artifact_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in required if path.is_file()
        },
    }
    (PAPER / "validation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if (
        missing_citations
        or missing_artifacts
        or not summary_digest_matches
        or pdf_pages != 4
        or not references_page_only
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
