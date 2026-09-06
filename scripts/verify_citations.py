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

  ATTESTED    no machine-resolvable record, but the author supplied a link to
              the paper. Recorded as author attestation, kept distinct from
              VERIFIED: it is evidence a human checked, not a publisher-record
              match, and it says nothing about the entry's year, venue or pages.
  ACCEPTED    weaker still: no record, no link -- the author accepted the entry
              as correct from their own knowledge of it. The one entry in this
              bibliography that earns it is OpenReview-only, so there is nothing
              to resolve and nothing to link that is not already the paper. It
              is labelled rather than quietly promoted, because the difference
              between "a human confirmed this against a record" and "a human
              says this is right" is the whole point of keeping the buckets
              apart.
  VERIFIED    the record exists and its title matches what was reported
  EQUIVALENT  the record exists and its title matches once the things a
              publisher adds but a bibliography does not are removed: a
              trailing parenthetical annotation, a spelled-out part label.
              Kept distinct from VERIFIED so that every entry which needed
              the weaker comparison is visible rather than folded in.
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


_PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*$")
_PART_LABEL = re.compile(r"\b(?:part|pt)\b")


def normalise(t):
    """Compare titles on words, not whitespace or punctuation.

    Punctuation is replaced by a space and the whitespace collapsed after, not
    before: the original did it the other way round, so an em dash became a
    doubled space and "Network Optimization -- Using Relays as Neurons" failed
    to match the same title written without the dash. Replacing rather than
    deleting also keeps "channels. II." from becoming "channelsii".
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (t or "").lower())).strip()


def title_relation(reported, actual):
    """How a record's title relates to the reported one.

    Returns ("exact", None) when the two normalise to the same words,
    ("equivalent", rules) when they do so only after removing what a publisher
    record carries and a bibliography does not, and (None, None) otherwise.

    Two such differences show up in this bibliography. IEEE files the BCJR
    paper as "... minimizing symbol error rate (Corresp.)", a record annotation
    rather than part of the title; and it files Hanly & Tse's "Multiaccess
    fading channels---part II: ..." as "Multiaccess fading channels. II. ...",
    dropping the word "part". Each rule is applied to both titles, so neither
    can make two different papers match unless they differ by nothing else.
    """
    if normalise(reported) == normalise(actual):
        return "exact", None
    rules = []
    a, b = (reported or "").strip(), (actual or "").strip()
    if _PARENTHETICAL.search(a) or _PARENTHETICAL.search(b):
        a, b = _PARENTHETICAL.sub("", a), _PARENTHETICAL.sub("", b)
        rules.append("a trailing parenthetical annotation")
    a, b = normalise(a), normalise(b)
    if a != b:
        a2, b2 = (re.sub(r"\s+", " ", _PART_LABEL.sub(" ", x)).strip()
                  for x in (a, b))
        if a2 != b2:
            return None, None
        rules.append("a spelled-out part label")
    return "equivalent", " and ".join(rules)


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
    # a resolvable identifier always wins: attestation is the fallback for
    # entries no index can confirm, never a shortcut past one that can
    pinned = cand.get("arxiv_id") or cand.get("doi")
    if cand.get("attested_url") and not pinned:
        return {"bibkey": cand.get("bibkey"), "reported_title": reported,
                "route": "author-attestation", "verdict": "ATTESTED",
                "url": cand["attested_url"],
                "reason": "author supplied a link to the paper; not a "
                          "publisher-record match"}
    if cand.get("accepted_by_author") and not pinned:
        return {"bibkey": cand.get("bibkey"), "reported_title": reported,
                "route": "author-acceptance", "verdict": "ACCEPTED",
                "url": None,
                "reason": "author accepted the entry as correct; no record "
                          "resolved and no link supplied, so this is weaker "
                          "than attestation: "
                          + str(cand.get("accepted_by_author"))}
    row = {"bibkey": cand.get("bibkey"),
           "arxiv_id": cand.get("arxiv_id"), "doi": cand.get("doi"),
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
            rec, relation, rules = None, None, None
            # exact first, over the whole result set: an equivalent match
            # further down the list must never displace an exact one
            for want in ("exact", "equivalent"):
                for it in items:
                    cand_rec = _crossref_record(it)
                    rel, rl = title_relation(reported, cand_rec["title"])
                    if rel == want:
                        rec, relation, rules = cand_rec, rel, rl
                        break
                if rec is not None:
                    break
            if rec is None:
                # No title match. Reporting the top hit as a MISMATCH would
                # accuse a possibly-correct reference of being a different
                # paper; a title search cannot support that claim. MISMATCH is
                # reserved for a pinned identifier resolving elsewhere. This is
                # an unresolved lookup, i.e. a FAIL. The nearest hit's DOI goes
                # in the record so the entry can be pinned without re-running
                # the search by hand.
                nearest = _crossref_record(items[0]) if items else None
                near = nearest["title"] if nearest else "--"
                near_doi = (nearest["id"] if nearest else None)
                row.update(verdict="FAIL", nearest_title=near,
                           nearest_url=near_doi,
                           reason="no record with this title in the top 10 "
                                  f"Crossref results (nearest: {near!r} at "
                                  f"{near_doi}); supply a DOI to pin it")
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
    relation, rules = title_relation(reported, rec["title"])
    searched = row["route"] == "crossref-title"
    caveat = ("" if not searched else
              "; found by title search, so this confirms existence, not the "
              "reference's other fields")
    if relation == "exact":
        row.update(verdict="VERIFIED", reason="title matches the record" + caveat)
    elif relation == "equivalent":
        row.update(verdict="EQUIVALENT", title_normalisation=rules,
                   reason=f"title matches the record once {rules} is removed"
                          + caveat)
    else:
        row.update(verdict="MISMATCH",
                   reason="record exists but the title differs from what was "
                          "reported; treat as FAIL pending manual check")
    return row


BEGIN, END = "<!-- VERIFICATION:BEGIN -->", "<!-- VERIFICATION:END -->"


def render_into_log(path, out):
    """Replace the marked block in the markdown log. Idempotent by design:
    the block is regenerated whole, so reruns do not stack entries."""
    icon = {"MISMATCH": "MISMATCH (treat as FAIL)"}
    c = out["counts"]
    L = [BEGIN,
         f"_Checked {out['checked_at']}. Standard: {out['standard']}._",
         "",
         f"**{c['VERIFIED']} verified, {c.get('EQUIVALENT', 0)} equivalent, "
         f"{c.get('ATTESTED', 0)} attested, {c.get('ACCEPTED', 0)} accepted, "
         f"{c['MISMATCH']} mismatched, "
         f"{c['FAIL']} failed.**",
         "",
         "| arXiv | Verdict | Title on the record | Published |",
         "|---|---|---|---|"]
    for r in out["results"]:
        title = r.get("actual_title") or "--"
        if len(title) > 90:
            title = title[:87] + "..."
        pub = (r.get("published") or "--")[:10]
        ident = r.get("arxiv_id") or r.get("doi") or r.get("bibkey") or "--"
        verdict = icon.get(r["verdict"], r["verdict"])
        L.append(f"| {ident} | {verdict} | {title} | {pub} |")
    bad = [r for r in out["results"]
           if r["verdict"] not in ("VERIFIED", "EQUIVALENT", "ATTESTED",
                                   "ACCEPTED")]
    if bad:
        L += ["", "Not citable:"]
        for r in bad:
            ident = r.get("arxiv_id") or r.get("doi") or r.get("bibkey") or "--"
            L.append(f"- **{ident}** -- {r['reason']}")
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
              for v in ("VERIFIED", "EQUIVALENT", "ATTESTED", "ACCEPTED",
                        "MISMATCH", "FAIL")}
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

    print(f"\n  {counts['VERIFIED']} verified, {counts['EQUIVALENT']} equivalent, "
          f"{counts['ATTESTED']} attested, {counts['ACCEPTED']} accepted, "
          f"{counts['MISMATCH']} mismatched, {counts['FAIL']} failed -> {a.out}")
    # A citable source is the point; a run that resolves nothing has failed at
    # its job even though every lookup "worked".
    return 0 if (counts["VERIFIED"] or counts["EQUIVALENT"]
                 or counts["ATTESTED"] or counts["ACCEPTED"]) else 1


if __name__ == "__main__":
    sys.exit(main())
