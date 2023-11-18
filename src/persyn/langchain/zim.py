"""Util that calls Zim."""
import logging
import re

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
import libzim

from ftfy import fix_text
from pydantic import BaseModel, Extra, root_validator
from bs4 import BeautifulSoup
from langchain.schema import Document

logger = logging.getLogger(__name__)

ZIM_MAX_QUERY_LENGTH = 300

class ZimWrapper(BaseModel):
    """Wrapper around libzim.

    To use, you should have the ``libzim`` and ``ftfy`` python packages installed.
    This wrapper will use libzim to conduct searches and fetch page summaries.
    By default, it will return the page summaries of the top-k results.
    It limits the Document content by doc_content_chars_max.

    To initialize, pass the path to a ZIM file:

        ZimWrapper(path="/home/ec2-user/wikipedia_en_all_nopic.zim")

    ...or the URL to the front page of a Kiwix instance:

        ZimWrapper(path="http://192.168.0.101:9999/viewer#wikipedia_en_all_maxi/")
    """

    path: str
    zim: Any  #: :meta private:
    server: Any  #: :meta private:
    search_url: Any  #: :meta private:
    content_url: Any  #: :meta private:
    top_k_results: int = 3
    lang: str = "en"
    load_all_available_meta: bool = False
    doc_content_chars_max: int = 4000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.path = kwargs['path']
        if self.path.startswith('http'):
            self.zim = None
            url = urlparse(self.path)
            book = url.fragment.split('/')[0]
            self.server = f'{url.scheme}://{url.netloc}'
            self.search_url =f'{self.server}/search?books.name={book}&pattern='
            self.content_url =f'{self.server}/content/{book}'

        else:
            self.zim = libzim.Archive(self.path)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            import libzim # pylint:disable=import-outside-toplevel
            import ftfy # pylint:disable=import-outside-toplevel

        except ImportError as ex:
            raise ImportError(
                "Could not import the libzim or ftfy python packages. "
                "Please install them with `pip install libzim ftfy`."
            ) from ex
        return values

    def run(self, query: str) -> str:
        """Run Zim search and get page summaries."""
        if self.zim:
            query = libzim.Query().set_query(query)
            searcher = libzim.Searcher(self.zim)
            search = searcher.search(query)
            page_titles = list(search.getResults(0, ZIM_MAX_QUERY_LENGTH))
        else:
            page_titles = self._web_search_results(query, top=self.top_k_results)
        summaries = []
        for page_title in page_titles[: self.top_k_results]:
            if wiki_page := self._fetch_page(page_title):
                if wiki_page:
                    summaries.append(wiki_page)
        if not summaries:
            return "No good Zim Search Result was found"
        return "\n\n".join(summaries)[: self.doc_content_chars_max]

    def _web_search_results(self, query, top=5):
        ''' Return kiwix search results for query '''
        web = requests.get(f"{self.search_url}{query}", timeout=10)
        web.raise_for_status()

        soup = BeautifulSoup(web.text, features="lxml")
        ret = []
        for elm in soup.select_one('.results').select('li')[:top]:
            href = elm.select_one('a')
            # Just a list of page names
            ret.append(href.attrs['href'].split('/')[-1])

        return ret

    @staticmethod
    def _cleanup(text: str) -> str:
        # Filter: no references
        norefs = re.sub(r'\[\d+\]', '', fix_text(text.strip()))
        # no phonetic spelling
        norefs = re.sub(r'\(/.*\(listen\)\) ?', '', norefs)
        # squash newlines
        return re.sub(r'\n+', '\n\n', norefs)

    def _fetch_page(self, page: str) -> Optional[str]:
        if not page:
            return None

        if self.zim:
            try:
                entry = self.zim.get_entry_by_path(page)
            except KeyError:
                return None
        else:
            web = requests.get(f"{self.content_url}/{page}", timeout=10)
            if web.status_code != 200:
                return None
            entry = web.text

        ret = []
        if self.zim:
            soup = BeautifulSoup(bytes(entry.get_item().content).decode("UTF-8"), features="lxml")
        else:
            soup = BeautifulSoup(entry, features="lxml")

        # (╯°□°)╯︵ ┻━┻
        for table in soup.select('table'):
            table.decompose()

        # title and summary
        ret.append(soup.select_one('.mw-headline').text)
        ret.append(soup.select_one('.mf-section-0').text)

        # remaining content
        for elm in soup.find_all('details'):
            # Skip the ending
            if elm.find('summary').text in ['See also', 'References', 'Further reading']:
                break

            ret.append(elm.text)
            ret.append('\n')

        return self._cleanup('\n'.join(ret))

    def _page_to_document(self, page_title: str, wiki_page: Any) -> Document:
        raise NotImplementedError('Not yet implemented')

    def load(self, query: str) -> List[Document]:
        raise NotImplementedError('Not yet implemented')
