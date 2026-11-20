# Website bibliography

This directory maintains Rahul Kakodkar's public bibliography. It preserves the
51 entries imported from the existing BibBase source, enriches identified papers,
and produces a BibTeX file for the current Google Sites/BibBase embed.

## Routine

A weekly ChatGPT automation checks ORCID and publisher records, runs the sync,
and proposes changes in a GitHub pull request. Rahul reviews and merges the PR.
GitHub Actions validates the bibliography and attaches the HTML preview on PRs
and pushes. The automation is the scheduler; this repository deliberately has
no second scheduled import job.

The routine does not merge its own PR, publish Google Sites, delete legacy
entries, or change ORCID account settings. New records must pass author identity
checks. ORCID alone is insufficient: one existing record has Rahul's ORCID
attached to another author's name in Crossref. That DOI is explicitly excluded
pending owner review. No author-name-only literature search is imported blindly.

Papers absent from ORCID can be reviewed and added to `additional_dois` in
`config.json`. To improve upstream discovery, use your ORCID at submission and
authorize Crossref/DataCite updates in ORCID; this setup does not assume those
account permissions have been enabled.

## Files

| File | Purpose |
| --- | --- |
| `source.bib` | Preserved original 51 records. Add manually curated talks/posters here. |
| `discovered.json` | Additional verified papers, with stable DOI-derived citation keys. |
| `metadata.json` | Publisher metadata and verified PDF links, keyed by citation key. |
| `overrides.json` | Durable corrections and manually selected links; always applied last. |
| `pdf-candidates.json` | Publisher/repository PDF URLs to check. |
| `config.json` | ORCID, identity checks, explicit DOI additions/exclusions, minimum count. |
| `publications.bib` | Generated BibBase source; do not edit directly. |
| `preview.html` | Generated searchable preview with type/year filters. |
| `preview-template.html` | Presentation template for the preview. |
| `embed.html` | Proposed Google Sites embed, to apply only after review. |
| `sync-report.json` | Items imported, matched, flagged, and PDF verification results. |

All source material here is public bibliographic metadata. PDFs are linked at
the publisher or repository; PDF files are not copied into this repository.

## Run locally

```bash
python3 -m venv .venv-bibliography
.venv-bibliography/bin/python -m pip install -r bibliography/requirements.txt
.venv-bibliography/bin/python scripts/bibliography.py sync
.venv-bibliography/bin/python -m unittest discover -s tests -v
.venv-bibliography/bin/python scripts/bibliography.py check
```

`build` regenerates the `.bib` and HTML without network access. `sync` reads the
public ORCID API, fetches metadata for known/new DOIs, verifies PDF responses,
and then builds both outputs. Existing years are retained and differing
publisher years are flagged because online/print publication dates can differ.
An upstream failure never replaces the bibliography with an empty result.
Temporary PDF failures retain previously verified links and appear in the
report. New PDF links must actually return PDF bytes; login pages are rejected.

## Corrections and future code links

Use the existing citation key in `overrides.json`, with `fields`, `reason`, and
`sources`. For example, set `fields.url_code` to the specific public repository
for that paper when it is ready. `url_pdf`, `url_data`, and `url_slides` are also
supported. Use `null` to explicitly suppress a field. Never publish private
manuscript or code URLs. No code links are populated in this first pass.

## Connect the live bibliography after review

1. Merge the setup PR so this URL exists:
   `https://raw.githubusercontent.com/cacodcar/cacodcar.github.io/master/bibliography/publications.bib`
2. Replace the source URL in the current BibBase embed with that URL. The
   proposed styling/embed is in `embed.html`; review it in Google Sites preview.
3. Publish Google Sites only after Rahul approves the preview.

Later approved `.bib` updates are reflected when BibBase refreshes its cached
source; the Google Sites page does not need republishing for each new entry.
The live Google Sites page remains on its current source until the one-time
switch is approved.

## Initial review notes

- The original BibBase source is recorded in `config.json`.
- The imported `Kakaodkar` spelling and hydrogen title suffix are corrected with
  persistent overrides. A corrupted Crossref dash is also prevented from
  replacing the readable title.
- DOI `10.1016/j.apenergy.2024.124900` is excluded: the publisher author list does
  not include Rahul, despite the ORCID association. Do not remove it from ORCID
  automatically; it is an account correction for Rahul to review.
- Papers without usable public PDF responses retain DOI/publisher links. An
  absent PDF button means no direct PDF was verified by this routine.
