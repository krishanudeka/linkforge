from data_pipeline.scrapers.base_scraper import BaseScraper, PaperRecord
from data_pipeline.scrapers.europe_pmc_scraper import EuropePMCScraper
from data_pipeline.scrapers.pubmed_scraper import PubMedScraper
from data_pipeline.scrapers.semantic_scholar import SemanticScholarScraper

SCRAPERS = {
    "europe_pmc": EuropePMCScraper,
    "pubmed": PubMedScraper,
    "semantic_scholar": SemanticScholarScraper,
}


def get_scraper(name: str) -> BaseScraper:
    try:
        return SCRAPERS[name.lower()]()
    except KeyError:
        raise ValueError(f"Unknown source '{name}'. Choose from: {', '.join(SCRAPERS)}")


__all__ = ["BaseScraper", "PaperRecord", "EuropePMCScraper", "PubMedScraper",
           "SemanticScholarScraper", "SCRAPERS", "get_scraper"]
