"""
Capa de soporte (tokenización, segmentación de oraciones, utilidades).
"""

from src.support.sentence_splitter import SentenceSplitter
from src.support.tokenizer import Tokenizer, TokenizerFactory

__all__ = ["SentenceSplitter", "Tokenizer", "TokenizerFactory"]
