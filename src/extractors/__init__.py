"""
Extractores de texto por formato de archivo (patrón Factory Method).
"""

from src.extractors.base import BaseExtractor, ExtractorError
from src.extractors.factory import ExtractorFactory, register_extractor
from src.extractors.csv_xlsx_extractor import CSVExtractor, XLSXExtractor
from src.extractors.html_extractor import HTMLExtractor
from src.extractors.image_extractor import ImageExtractor
from src.extractors.json_extractor import JSONExtractor
from src.extractors.md_txt_extractor import MarkdownTxtExtractor
from src.extractors.pbf_extractor import PBFExtractor
from src.extractors.pdf_extractor import PDFExtractor

__all__ = [
    "BaseExtractor",
    "CSVExtractor",
    "ExtractorError",
    "ExtractorFactory",
    "HTMLExtractor",
    "ImageExtractor",
    "JSONExtractor",
    "MarkdownTxtExtractor",
    "PBFExtractor",
    "PDFExtractor",
    "XLSXExtractor",
    "register_extractor",
]
