#!/usr/bin/env python3
"""Preserve a curated bibliography, enrich verified works, and build reviewable output."""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import bibtexparser
from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.latexencode import unicode_to_latex

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'bibliography'
USER_AGENT = 'KakodkarBibliography/1.0 (+https://www.cacodcar.com)'
LATEX = LatexNodes2Text()


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_if_changed(path: Path, text: str):
    if path.exists() and path.read_text(encoding='utf-8') == text:
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def save_json(path: Path, value):
    write_if_changed(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def plain(value: str) -> str:
    return LATEX.latex_to_text(value)


def normalized(value: str) -> str:
    value = unicodedata.normalize('NFKD', plain(value)).casefold()
    return ''.join(c for c in value if c.isalnum())


def doi_value(value: str) -> str:
    return re.sub(r'^(https?://(dx\.)?doi\.org/|doi:\s*)', '', value.strip(), flags=re.I).lower()


def safe_url(value: str) -> bool:
    parsed = urllib.parse.urlsplit(value)
    return parsed.scheme in ('https', 'http') and bool(parsed.hostname) and not parsed.username and not parsed.password


def request(url: str, accept: str = 'application/json'):
    if not safe_url(url):
        raise ValueError(f'Unsupported URL: {url}')
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept': accept})
    return urllib.request.urlopen(req, timeout=30)


def get_json(url: str):
    # Honor a numeric Retry-After once; long waits are left to the next run.
    try:
        with request(url) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        delay = error.headers.get('Retry-After', '')
        if error.code == 429 and delay.isdigit() and 0 < int(delay) <= 30:
            time.sleep(int(delay))
            with request(url) as response:
                return json.load(response)
        raise


def read_source(path: Path) -> dict:
    raw = path.read_text(encoding='utf-8')
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    db = bibtexparser.loads(raw, parser=parser)
    expected = len(re.findall(r'^\s*@(?!(?:comment|string|preamble)\b)\w+\s*[{(]', raw, re.I | re.M))
    entries = {entry['ID']: entry for entry in db.entries}
    if len(entries) != len(db.entries) or len(entries) != expected:
        raise ValueError('Duplicate citation keys or unparsed BibTeX entries; refusing partial output.')
    return entries


def combined(data: Path) -> dict:
    entries = read_source(data / 'source.bib')
    discovered = load_json(data / 'discovered.json')
    if entries.keys() & discovered.keys():
        raise ValueError('A discovered key collides with the preserved source.')
    entries.update(copy.deepcopy(discovered))
    metadata = load_json(data / 'metadata.json')
    overrides = load_json(data / 'overrides.json')
    for key in metadata.keys() | overrides.keys():
        if key not in entries:
            raise ValueError(f'Unknown citation key: {key}')
        for layer in (metadata.get(key, {}), overrides.get(key, {})):
            for field, value in layer.get('fields', {}).items():
                if field == 'ID':
                    raise ValueError('Citation keys cannot be overridden.')
                if value is None:
                    entries[key].pop(field, None)
                else:
                    entries[key][field] = value
    return entries


def validate(entries: dict, minimum: int = 0):
    if len(entries) < minimum:
        raise ValueError(f'Expected at least {minimum} entries, found {len(entries)}.')
    seen = {}
    for key, entry in entries.items():
        if entry.get('ID') != key or not entry.get('title') or not entry.get('author'):
            raise ValueError(f'Invalid entry: {key}')
        for field, value in entry.items():
            if field.startswith('url') and not safe_url(value):
                raise ValueError(f'Unsafe link in {key}: {field}')
        doi = doi_value(entry.get('doi', ''))
        if doi:
            if not re.fullmatch(r'10\.\d{4,9}/\S+', doi):
                raise ValueError(f'Invalid DOI in {key}')
            if doi in seen:
                raise ValueError(f'Duplicate DOI in {seen[doi]} and {key}')
            seen[doi] = key


def author_matches(work: dict, config: dict) -> bool:
    for author in work.get('author', []):
        orcid = author.get('ORCID', '').rstrip('/').rsplit('/', 1)[-1]
        family = normalized(author.get('family', ''))
        given = normalized(author.get('given', ''))
        if family in config['family_names']:
            if given == config['given_name']:
                return True
            if orcid == config['orcid'] and given == config['given_name'][0]:
                return True
    return False


def title_match(entries: dict, title: str, year: str) -> str | None:
    # No fuzzy merging: talks and articles often share a similar title.
    matches = [key for key, entry in entries.items()
               if normalized(entry['title']) == normalized(title)
               and entry.get('year') == year
               and entry['ENTRYTYPE'] not in ('misc', 'mastersthesis', 'phdthesis')
               and 'aiche annual meeting' not in entry.get('booktitle', '').lower()]
    return matches[0] if len(matches) == 1 else None


def tex(value: str) -> str:
    value = html.unescape(re.sub(r'<[^>]+>', '', value))
    return unicode_to_latex(value, non_ascii_only=False)


def crossref_fields(work: dict, existing_type: str | None = None) -> dict:
    kind = existing_type or {'journal-article': 'article', 'book-chapter': 'incollection',
                             'proceedings-article': 'inproceedings'}.get(work.get('type'), 'misc')
    fields = {'ENTRYTYPE': kind, 'doi': doi_value(work['DOI']), 'url': 'https://doi.org/' + doi_value(work['DOI'])}
    if work.get('title'):
        fields['title'] = '{' + tex(work['title'][0]) + '}'
    authors = ['{}, {}'.format(tex(a['family']), tex(a.get('given', ''))).rstrip(', ')
               if a.get('family') else '{' + tex(a['name']) + '}'
               for a in work.get('author', []) if a.get('family') or a.get('name')]
    if authors:
        fields['author'] = ' and '.join(authors)
    if work.get('container-title'):
        fields['journal' if kind == 'article' else 'booktitle'] = tex(work['container-title'][0])
    for source, target in [('volume', 'volume'), ('issue', 'number'), ('page', 'pages'), ('publisher', 'publisher')]:
        if work.get(source):
            fields[target] = tex(str(work[source]))
    if not fields.get('pages') and work.get('article-number'):
        fields['pages'] = str(work['article-number'])
    return fields


def publication_year(work: dict) -> str:
    for field in ('published-print', 'published', 'published-online', 'issued'):
        parts = work.get(field, {}).get('date-parts', [[]])
        if parts and parts[0]:
            return str(parts[0][0])
    raise ValueError('Publication has no year.')


def verify_pdf(url: str) -> bool:
    try:
        with request(url, '*/*') as response:
            return safe_url(response.url) and response.read(1024).lstrip().startswith(b'%PDF-')
    except (OSError, ValueError):
        return False


def sync(data: Path, cache: Path | None = None):
    config = load_json(data / 'config.json')
    entries = combined(data)
    validate(entries, config['minimum_entries'])
    metadata = load_json(data / 'metadata.json')
    discovered = load_json(data / 'discovered.json')
    candidates = load_json(data / 'pdf-candidates.json')
    report = {'flagged': [], 'imported': [], 'matched': [], 'pdf_verified': [], 'pdf_unavailable': []}
    works_url = f'https://pub.orcid.org/v3.0/{config["orcid"]}/works'
    orcid = load_json(cache / 'orcid-works.json') if cache else get_json(works_url)
    cached_crossref = load_json(cache / 'crossref.json') if cache else {}
    doi_to_key = {doi_value(e['doi']): key for key, e in entries.items() if e.get('doi')}
    incoming = set(doi_to_key) | {doi_value(d) for d in config.get("additional_dois", [])}
    for group in orcid.get('group', []):
        identifiers = group.get('external-ids', {}).get('external-id', [])
        dois = {doi_value(e['external-id-value']) for e in identifiers
                if e['external-id-type'].lower() == 'doi' and e.get('external-id-relationship') == 'self'}
        if len(dois) == 1:
            incoming.update(dois)
        elif len(dois) > 1:
            report['flagged'].append({'dois': sorted(dois), 'reason': 'Multiple self DOIs in one ORCID group; review versions.'})
        else:
            summary = group.get('work-summary', [{}])[0]
            report['flagged'].append({'title': summary.get('title', {}).get('title', {}).get('value'),
                                      'reason': 'No DOI in ORCID; existing entries preserved.'})
    for doi in sorted(incoming):
        if doi in config.get('ignored_dois', {}):
            report['flagged'].append({'doi': doi, 'reason': config['ignored_dois'][doi]})
            continue
        url = 'https://api.crossref.org/works/' + urllib.parse.quote(doi, safe='/')
        try:
            if cache:
                work = cached_crossref[doi]
            else:
                work = get_json(url)['message']
                time.sleep(1)
        except (OSError, KeyError, ValueError) as error:
            report['flagged'].append({'doi': doi, 'reason': f'Metadata fetch failed: {error}'})
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                raise RuntimeError('Crossref rate limit; no sync state has been written. Retry later.') from error
            continue
        key = doi_to_key.get(doi)
        if not key and not author_matches(work, config):
            report['flagged'].append({'doi': doi, 'reason': 'Publisher author identity does not match; not imported.'})
            continue
        year = publication_year(work)
        title = work.get('title', [''])[0]
        if not key:
            key = title_match(entries, title, year)
        if key:
            existing_doi = doi_value(entries[key].get('doi', ''))
            if existing_doi and existing_doi != doi:
                report['flagged'].append({'doi': doi, 'reason': 'Matching title has a different DOI; review versions.'})
                continue
            report['matched'].append(key)
            if year != entries[key].get('year'):
                report['flagged'].append({'key': key, 'doi': doi, 'reason': f'Existing year {entries[key].get("year")} retained; publisher year is {year}.'})
        else:
            # A stable DOI-based key survives future title or date corrections.
            key = 'doi_' + hashlib.sha256(doi.encode()).hexdigest()[:12]
            if key in entries:
                raise ValueError('DOI key collision.')
            fields = crossref_fields(work)
            fields.update(ID=key, year=year)
            if not fields.get('author') or not fields.get('title'):
                report['flagged'].append({'doi': doi, 'reason': 'Missing title or authors.'})
                continue
            discovered[key] = fields
            entries[key] = copy.deepcopy(fields)
            report['imported'].append(key)
        doi_to_key[doi] = key
        fields = crossref_fields(work, entries[key]['ENTRYTYPE'])
        previous = metadata.get(key, {}).get('fields', {})
        # Preserve hand-added resource links; automatic DOI refresh cannot erase them.
        for field, value in previous.items():
            if field.startswith('url_'):
                fields[field] = value
        metadata[key] = {'fields': fields, 'source': url}
        pdfs = list(candidates.get(key, []))
        pdfs += [link['URL'] for link in work.get('link', []) if link.get('content-type') == 'application/pdf']
        candidates[key] = list(dict.fromkeys(pdfs))
    for key, urls in candidates.items():
        if key not in entries:
            raise ValueError(f'PDF candidate has unknown key: {key}')
        existing = metadata.get(key, {}).get('fields', {}).get('url_pdf')
        for url in dict.fromkeys(([existing] if existing else []) + urls):
            if verify_pdf(url):
                layer = metadata.setdefault(key, {'fields': {}})
                layer['fields']['url_pdf'] = url
                layer['pdf_source'] = url
                report['pdf_verified'].append(key)
                break
        else:
            if urls:
                report['pdf_unavailable'].append(key)
    # Validate the complete proposed dataset before any persistent sync state is written.
    proposed = copy.deepcopy(entries)
    overrides = load_json(data / 'overrides.json')
    for key, layer in metadata.items():
        proposed[key].update(layer['fields'])
    for key, layer in overrides.items():
        for field, value in layer.get('fields', {}).items():
            if value is None:
                proposed[key].pop(field, None)
            else:
                proposed[key][field] = value
    validate(proposed, config['minimum_entries'])
    save_json(data / 'metadata.json', metadata)
    save_json(data / 'discovered.json', discovered)
    save_json(data / 'pdf-candidates.json', candidates)
    save_json(data / 'sync-report.json', report)
    return report


def bib_text(entries: dict) -> str:
    db = bibtexparser.bibdatabase.BibDatabase()
    db.entries = list(entries.values())
    writer = bibtexparser.bwriter.BibTexWriter()
    writer.indent = '  '
    writer.order_entries_by = ('ID',)
    return bibtexparser.dumps(db, writer=writer)


def category(entry: dict) -> str:
    note = entry.get('note', '').lower()
    venue = (entry.get('journal', '') + ' ' + entry.get('booktitle', '')).lower()
    if 'poster' in note:
        return 'Posters'
    if 'oral presentation' in note or 'aiche annual meeting' in venue:
        return 'Talks'
    if entry['ENTRYTYPE'] in ('mastersthesis', 'phdthesis'):
        return 'Theses'
    if any(name in venue for name in ('computer aided chemical engineering', 'systems and control transactions', 'ifac-papersonline', 'foundations of computer-aided process operations')):
        return 'Proceedings'
    return {'article': 'Journal articles', 'incollection': 'Chapters', 'inproceedings': 'Proceedings'}.get(entry['ENTRYTYPE'], 'Other')


def build(data: Path):
    entries = combined(data)
    config = load_json(data / 'config.json')
    validate(entries, config['minimum_entries'])
    serialized = bib_text(entries)
    roundtrip = bibtexparser.loads(serialized).entries
    if len(roundtrip) != len(entries):
        raise ValueError('BibTeX round-trip lost entries.')
    write_if_changed(data / 'publications.bib', serialized)
    rows = []
    for key, e in sorted(entries.items(), key=lambda kv: (-int(kv[1].get('year', '0')), plain(kv[1]['title']).lower())):
        links = []
        if e.get('doi'):
            links.append(('DOI', 'https://doi.org/' + doi_value(e['doi'])))
        if e.get('url') and (not e.get('doi') or doi_value(e['url']) != doi_value(e['doi'])):
            links.append(('Paper', e['url']))
        for field, label in [('url_pdf', 'PDF'), ('url_code', 'Code'), ('url_data', 'Data'), ('url_slides', 'Slides')]:
            if e.get(field):
                links.append((label, e[field]))
        venue = plain(e.get('journal', e.get('booktitle', e.get('school', ''))))
        details = ' · '.join(filter(None, [venue, e.get('year'), e.get('volume'), e.get('pages')]))
        authors = plain(e['author']).replace(' and ', '; ')
        title = plain(e['title'])
        group = category(e)
        actions = ' '.join(f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">{label} ↗</a>' for label, url in links)
        rows.append(f'''<article data-year="{html.escape(e.get('year', ''))}" data-type="{html.escape(group)}">
<div class="eyebrow">{html.escape(group)} <span>{html.escape(e.get('year', ''))}</span></div>
<h2>{html.escape(title)}</h2><p class="authors">{html.escape(authors)}</p>
<p class="venue">{html.escape(details)}</p><p class="note">{html.escape(plain(e.get('note', '')))}</p>
<div class="actions">{actions}<details><summary>BibTeX</summary><pre>{html.escape(bib_text({key:e}))}</pre></details></div></article>''')
    template = (data / 'preview-template.html').read_text(encoding='utf-8')
    template = template.replace('<!-- ENTRIES -->', '\n'.join(rows)).replace('{{COUNT}}', str(len(entries)))
    years = sorted({e.get('year', '') for e in entries.values()}, reverse=True)
    types = sorted({category(e) for e in entries.values()})
    for marker, values in [('<!-- YEARS -->', years), ('<!-- TYPES -->', types)]:
        template = template.replace(marker, ''.join(f'<option>{html.escape(v)}</option>' for v in values))
    write_if_changed(data / 'preview.html', template)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'sync', 'check'])
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--cache', type=Path, help='Use saved ORCID/Crossref responses (PDF verification remains live).')
    args = parser.parse_args()
    if args.command == 'sync':
        print(json.dumps(sync(args.data_dir, args.cache), indent=2))
    if args.command in ('build', 'sync'):
        entries = build(args.data_dir)
    else:
        entries = combined(args.data_dir)
        validate(entries, load_json(args.data_dir / 'config.json')['minimum_entries'])
    print(f'Validated {len(entries)} entries; {sum(bool(e.get("doi")) for e in entries.values())} DOI links; '
          f'{sum(bool(e.get("url_pdf")) for e in entries.values())} PDF links.')


if __name__ == '__main__':
    main()
