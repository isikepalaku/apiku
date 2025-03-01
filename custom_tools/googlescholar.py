import json
from typing import Any, Dict, List, Optional

from agno.tools import Toolkit
from agno.utils.log import logger

try:
    from scholarly import scholarly
except ImportError:
    raise ImportError("`scholarly` tidak ditemukan. Silakan install dengan `pip install scholarly`")


class GoogleScholarTools(Toolkit):
    """
    GoogleScholarTools adalah library agno untuk mencari publikasi di Google Scholar.
    Library ini menggunakan paket scholarly untuk mengakses data Google Scholar.

    Args:
        fixed_max_results (Optional[int]): Jumlah maksimum hasil pencarian yang ditetapkan.
        headers (Optional[Any]): Header custom untuk request (jika diperlukan).
        proxy (Optional[str]): Proxy untuk request (jika diperlukan).
        timeout (Optional[int]): Timeout untuk request, default 10 detik.
    """

    def __init__(
        self,
        fixed_max_results: Optional[int] = None,
        headers: Optional[Any] = None,
        proxy: Optional[str] = None,
        timeout: Optional[int] = 10,
    ):
        super().__init__(name="googlescholar")
        self.fixed_max_results: Optional[int] = fixed_max_results
        self.headers: Optional[Any] = headers
        self.proxy: Optional[str] = proxy
        self.timeout: Optional[int] = timeout

        self.register(self.google_scholar_search)

    def google_scholar_search(self, query: str, max_results: int = 5) -> str:
        """
        Fungsi untuk melakukan pencarian di Google Scholar berdasarkan query yang diberikan.

        Args:
            query (str): Query pencarian.
            max_results (int, optional): Jumlah maksimum hasil pencarian. Default adalah 5.

        Returns:
            str: String JSON yang berisi hasil pencarian.
        """
        # Gunakan fixed_max_results jika telah diset
        max_results = self.fixed_max_results or max_results

        logger.debug(f"Melakukan pencarian di Google Scholar untuk: {query}")

        results: List[Dict[str, str]] = []
        # Melakukan iterasi terhadap hasil pencarian publikasi
        for idx, publication in enumerate(scholarly.search_pubs(query)):
            if idx >= max_results:
                break
            bib = publication.get("bib", {})
            results.append({
                "title": bib.get("title", ""),
                "authors": bib.get("author", ""),
                "year": bib.get("pub_year", ""),
                "venue": bib.get("venue", ""),
                "abstract": bib.get("abstract", ""),
                "url": publication.get("url_scholarbib", ""),
            })

        return json.dumps(results, indent=2)
