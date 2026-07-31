"""
Fixtures compartidas de los tests: segmentador y config de chunking.
"""

from __future__ import annotations

import pytest

from src.chunking.base import TextSegmenter
from src.models.config import ChunkingConfig
from src.support.sentence_splitter import SentenceSplitter
from src.support.tokenizer import Tokenizer


@pytest.fixture(scope="session")
def segmenter() -> TextSegmenter:
    """Segmentador determinista (regex) para tests rápidos y estables."""
    return TextSegmenter(
        tokenizer=Tokenizer("cl100k_base"),
        splitter=SentenceSplitter(),  # sin spacy: usa el segmentador regex
    )


@pytest.fixture
def chunking_config() -> ChunkingConfig:
    """Config de chunking por defecto para los tests."""
    return ChunkingConfig(
        chunk_size=400,
        overlap_size=80,
        min_chunk_tokens=50,
        max_tokens=512,
    )


@pytest.fixture
def chunking_config_pequeno() -> ChunkingConfig:
    """Config con ventanas pequeñas para ejercitar casos borde."""
    return ChunkingConfig(
        chunk_size=60,
        overlap_size=15,
        min_chunk_tokens=25,
        max_tokens=512,
    )
