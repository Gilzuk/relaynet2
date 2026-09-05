#!/usr/bin/env python3
"""Verify candidate arXiv sources against the arXiv API. Gray zone is a FAIL.

The literature phase of the BCJR/APP benchmark research could not be completed
from the development container: WebSearch worked, but arxiv.org, wikipedia,
api.semanticscholar.org and api.crossref.org are all refused by its egress
proxy, so no candidate could be checked against its own record. A search
result is not a citation. This script does the check somewhere with open
network access -- a CI runner -- and records the outcome.

The standard it applies (deep-research skill, IRON RULE #4): a source that
cannot be confirmed is a FAIL, not an "uncertain". Three verdicts:

  VERIFIED    the record exists and its title matches what was reported
  MISMATCH    the record exists but the title differs -- treat as FAIL. This
              is the signature of a fabricated or mashed-up reference, which
              is the hardest kind to catch by eye, so it is reported loudly
              rather than folded into VERIFIED.
  FAIL        no record, or the lookup itself did not succeed

Usage:
  python3 scripts/verify_citations.py \
      --candidates docs/research/bcjr-candidates.json \
      --out docs/research/bcjr-candidates-verified.json
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

API = "http://export.arxiv.org/api/query?id_list={}"
CROSSREF_DOI = "https://api.crossref.org/works/{}"
CROSSREF_QUERY = ("https://api.crossref.org/works?rows=10&select=title,author,"
                  "issued,container-title,DOI&query.bibliographic={}")
ATOM = "{http://www.w3.org/2005/Atom}"
UA = "relaynet2-citation-check (thesis research log; contact via repo issues)"


def normalise(t):
    """Compare titles on words, not whitespace or punctuation."""
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", (t or "").lower())).strip()


def fetch(arxiv_id, retries=3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API.format(arxiv_id),
                                         headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read()
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last = e
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"lookup failed after {retries} attempts: {last}")


def fetch_json(url, retries=3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
            last = e
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"lookup failed after {retries} attempts: {last}")


def _crossref_record(item):
    title = (item.get("title") or [""])[0]
    authors = [" ".join(x for x in (a.get("given"), a.get("family")) if x)
               for a in item.get("author", [])]
    year = None
    for k in ("issued", "published-print", "published-online"):
        parts = (item.get(k) or {}).get("date-parts") or [[]]
        if parts and parts[0]:
            year = parts[0][0]
            break
    return {"title": re.sub(r"\s+", " ", title).strip(),
            "authors": authors,
            "published": str(year) if year else None,
            "updated": None,
            "abstract": "",
            "id": "https://doi.org/" + item.get("DOI", "")}


def parse(xml_bytes):
    root = ET.fromstring(xml_bytes)
    entry = root.find(f"{ATOM}entry")
    if entry is None:
        return None
    # arXiv answers an unknown id with an entry whose id is the query echo
    title = entry.findtext(f"{ATOM}title")
    if title is None:
        return None
    summary = (entry.findtext(f"{ATOM}summary") or "").strip()
    return {
        "title": re.sub(r"\s+", " ", title).strip(),
        "authors": [a.findtext(f"{ATOM}name")
                    for a in entry.findall(f"{ATOM}author")],
        "published": entry.findtext(f"{ATOM}published"),
        "updated": entry.findtext(f"{ATOM}updated"),
        "abstract": re.sub(r"\s+", " ", summary),
        "id": entry.findtext(f"{ATOM}id"),
    }


def verify(cand):
    """Resolve a candidate against a publisher or preprint record.

    Three routes, in order of how much the identifier pins down: an arXiv id,
    a DOI, or -- for a reference carrying neither -- a Crossref bibliographic
    search on the reported title. The search route is the weakest: it confirms
    that a record with this title exists, not that the reference's other fields
    are right, and it is labelled as such in the reason.
    """
    reported = cand["reported_title"]
    cid = cand.get("arxiv_id") or cand.get("doi") or "(title search)"
    row = {"arxiv_id": cand.get("arxiv_id"), "doi": cand.get("doi"),
           "reported_title": reported, "route": None}
    try:
        if cand.get("arxiv_id"):
            row["route"] = "arxiv"
            rec = parse(fetch(cand["arxiv_id"]))
        elif cand.get("doi"):
            row["route"] = "crossref-doi"
            rec = _crossref_record(fetch_json(
                CROSSREF_DOI.format(urllib.parse.quote(cand["doi"])))["message"])
        else:
            row["route"] = "crossref-title"
            items = fetch_json(CROSSREF_QUERY.format(
                urllib.parse.quote(reported)))["message"]["items"]
            rec = None
            for it in items:
                cand_rec = _crossref_record(it)
                if normalise(cand_rec["title"]) == normalise(reported):
                    rec = cand_rec
                    break
            if rec is None:
                # No exact title match. Reporting the top hit as a MISMATCH
                # would accuse a possibly-correct reference of being a
                # different paper; a title search cannot support that claim.
                # MISMATCH is reserved for a pinned identifier resolving
                # elsewhere. This is an unresolved lookup, i.e. a FAIL.
                near = _crossref_record(items[0])["title"] if items else "--"
                row.update(verdict="FAIL",
                           reason="no record with this title in the top 10 "
                                  f"Crossref results (nearest: {near!r}); "
                                  "supply a DOI to pin it")
                return row
    except Exception as e:                       # network, parse, anything
        row.update(verdict="FAIL", reason=f"lookup error: {e}")
        return row
    if rec is None:
        row.update(verdict="FAIL", reason="no such record on arXiv")
        return row
    row.update(actual_title=rec["title"], authors=rec["authors"],
               published=rec["published"], updated=rec["updated"],
               abstract=rec["abstract"], url=rec["id"])
    if normalise(rec["title"]) == normalise(reported):
        note = ("title matches the record" if row["route"] != "crossref-title"
                else "title matches a Crossref record found by title search; "
                     "confirms existence, not the reference's other fields")
        row.update(verdict="VERIFIED", reason=note)
    else:
        row.update(verdict="MISMATCH",
                   reason="record exists but the title differs from what was "
                          "reported; treat as FAIL pending manual check")
    return row


BEGIN, END = "<!-- VERIFICATION:BEGIN -->", "<!-- VERIFICATION:END -->"


def render_into_log(path, out):
    """Replace the marked block in the markdown log. Idempotent by design:
    the block is regenerated whole, so reruns do not stack entries."""
    icon = {"VERIFIED": "VERIFIED", "MISMATCH": "MISMATCH (treat as FAIL)",
            "FAIL": "FAIL"}
    c = out["counts"]
    L = [BEGIN,
         f"_Checked {out['checked_at']}. Standard: {out['standard']}._",
         "",
         f"**{c['VERIFIED']} verified, {c['MISMATCH']} mismatched, "
         f"{c['FAIL']} failed.**",
         "",
         "| arXiv | Verdict | Title on the record | Published |",
         "|---|---|---|---|"]
    for r in out["results"]:
        title = r.get("actual_title") or "--"
        if len(title) > 90:
            title = title[:87] + "..."
        pub = (r.get("published") or "--")[:10]
        L.append(f"| {r['arxiv_id']} | {icon[r['verdict']]} | {title} | {pub} |")
    bad = [r for r in out["results"] if r["verdict"] != "VERIFIED"]
    if bad:
        L += ["", "Not citable:"]
        for r in bad:
            L.append(f"- **{r['arxiv_id']}** -- {r['reason']}")
    L += ["", "Full records, including authors and abstracts, are in "
              "`docs/research/bcjr-candidates-verified.json`.", END]
    block = "\n".join(L)

    body = open(path, encoding="utf-8").read()
    if BEGIN in body and END in body:
        pre, rest = body.split(BEGIN, 1)
        _, post = rest.split(END, 1)
        body = pre + block + post
    else:
        body = body.rstrip() + "\n\n## Verification results\n\n" + block + "\n"
    open(path, "w", encoding="utf-8").write(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", help="markdown log whose VERIFICATION block to replace")
    a = ap.parse_args()

    src = json.load(open(a.candidates))
    for c in src["candidates"]:
        if not (c.get("arxiv_id") or c.get("doi") or c.get("reported_title")):
            raise SystemExit(f"candidate has no identifier and no title: {c}")
    rows = []
    for c in src["candidates"]:
        label = (f"arXiv:{c['arxiv_id']}" if c.get("arxiv_id")
                 else f"doi:{c['doi']}" if c.get("doi")
                 else f"title:{c['reported_title'][:52]}")
        print(f"  checking {label} ...", flush=True)
        row = verify(c)
        print(f"    {row['verdict']}: {row['reason']}", flush=True)
        rows.append(row)
        time.sleep(3)                            # arXiv asks for >=3s between calls

    counts = {v: sum(1 for r in rows if r["verdict"] == v)
              for v in ("VERIFIED", "MISMATCH", "FAIL")}
    out = {"topic": src.get("topic"),
           "checked_at": datetime.now(timezone.utc).isoformat(),
           "standard": "deep-research IRON RULE #4 -- gray zone is a FAIL",
           "counts": counts, "results": rows}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    if a.log:
        render_into_log(a.log, out)
        print(f"  log block updated -> {a.log}")

    print(f"\n  {counts['VERIFIED']} verified, {counts['MISMATCH']} mismatched, "
          f"{counts['FAIL']} failed -> {a.out}")
    # A citable source is the point; a run that verifies nothing has failed at
    # its job even though every lookup "worked".
    return 0 if counts["VERIFIED"] else 1


if __name__ == "__main__":
    sys.exit(main())
