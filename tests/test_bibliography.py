"""Protect attribution, curated links, and the preservation of legacy records."""
import copy
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import bibliography as bib


class BibliographyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        self.entry = {'ID': 'paper', 'ENTRYTYPE': 'article', 'title': 'Energy & systems',
                      'author': 'Kakodkar, Rahul', 'year': '2024'}
        (self.data/'source.bib').write_text(bib.bib_text({'paper':self.entry}))
        for name in ['discovered', 'metadata', 'overrides', 'pdf-candidates']:
            bib.save_json(self.data/f'{name}.json', {})
        self.config = {'orcid':'0000-0002-4139-083X', 'family_names':['kakodkar','kakaodkar'],
                       'given_name':'rahul', 'minimum_entries':1}
        bib.save_json(self.data/'config.json', self.config)

    def test_curated_changes_and_code_links_survive_refresh(self):
        bib.save_json(self.data/'metadata.json', {'paper':{'fields':{'title':'Imported title', 'url_pdf':'https://example.org/paper.pdf'}}})
        bib.save_json(self.data/'overrides.json', {'paper':{'fields':{'title':'Corrected title', 'url_code':'https://github.com/example/code'}}})
        entry = bib.combined(self.data)['paper']
        self.assertEqual(entry['title'], 'Corrected title')
        self.assertEqual(entry['url_pdf'], 'https://example.org/paper.pdf')
        self.assertIn('url_code', entry)
        self.assertEqual(bib.read_source(self.data/'source.bib')['paper'], self.entry)

    def test_wrong_author_with_owners_orcid_is_not_attributed(self):
        work = {'author':[{'given':'Efstratios N.', 'family':'Pistikopoulos', 'ORCID':'https://orcid.org/'+self.config['orcid']}]}
        self.assertFalse(bib.author_matches(work, self.config))
        work['author'].append({'given':'Rahul','family':'Kakodkar'})
        self.assertTrue(bib.author_matches(work, self.config))

    def test_duplicate_doi_urls_are_rejected(self):
        a = dict(self.entry, doi='HTTPS://doi.org/10.1000/ABC')
        b = dict(self.entry, ID='other', doi='10.1000/abc')
        with self.assertRaisesRegex(ValueError, 'Duplicate DOI'):
            bib.validate({'paper':a,'other':b})

    def test_malformed_source_never_silently_drops_record(self):
        with (self.data/'source.bib').open('a') as f:
            f.write('\n@article{broken, title={Unclosed')
        with self.assertRaises(ValueError):
            bib.read_source(self.data/'source.bib')

    def test_title_match_keeps_talk_and_article_separate(self):
        talk = dict(self.entry, ID='talk', ENTRYTYPE='misc', note='Oral presentation')
        self.assertEqual(bib.title_match({'talk':talk,'paper':self.entry},'Energy & systems','2024'),'paper')
        self.assertIsNone(bib.title_match({'talk':talk},'Energy & systems','2024'))

    def test_network_failure_leaves_all_existing_files_untouched(self):
        before = {p.name:p.read_bytes() for p in self.data.iterdir()}
        with patch.object(bib,'get_json',side_effect=OSError('offline')):
            with self.assertRaises(OSError):
                bib.sync(self.data)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.data.iterdir()})

    def test_sync_preserves_talks_and_curated_fields(self):
        talk = dict(self.entry, ID='talk', ENTRYTYPE='misc', note='Oral presentation')
        (self.data/'source.bib').write_text(bib.bib_text({'paper':self.entry,'talk':talk}))
        bib.save_json(self.data/'overrides.json', {'paper':{'fields':{'url_code':'https://github.com/example/code'}}})
        payload = {'group':[{'external-ids':{'external-id':[{'external-id-type':'doi','external-id-value':'10.1000/test','external-id-relationship':'self'}]}}]}
        crossref = {'DOI':'10.1000/test','title':['Energy & systems'],'type':'journal-article',
                    'published':{'date-parts':[[2024]]},'author':[{'family':'Kakodkar','given':'Rahul'}]}
        with patch.object(bib,'get_json',side_effect=[payload,{'message':crossref}]), patch.object(bib.time,'sleep'):
            bib.sync(self.data)
        final = bib.combined(self.data)
        self.assertEqual(set(final),{'paper','talk'})
        self.assertEqual(final['talk'],talk)
        self.assertEqual(final['paper']['url_code'],'https://github.com/example/code')
        self.assertEqual(final['paper']['doi'],'10.1000/test')

    def test_pdf_endpoint_returning_login_html_is_rejected(self):
        @contextmanager
        def response(*args):
            data=BytesIO(b'<html>Please sign in</html>');data.url='https://example.org/login';yield data
        with patch.object(bib,'request',response):
            self.assertFalse(bib.verify_pdf('https://example.org/paper.pdf'))

    def test_unexpected_metadata_key_fails(self):
        bib.save_json(self.data/'metadata.json', {'unknown':{'fields':{'title':'Oops'}}})
        with self.assertRaisesRegex(ValueError,'Unknown citation key'):
            bib.combined(self.data)

    def test_unsafe_links_are_rejected(self):
        with self.assertRaises(ValueError):
            bib.validate({'paper':dict(self.entry,url_pdf='javascript:alert(1)')})


if __name__ == '__main__':
    unittest.main()
