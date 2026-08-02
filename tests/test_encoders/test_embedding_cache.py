"""
Tests: ``EmbeddingCache`` no debe marcar como pendiente un chunk ya
codificado con el mismo encoder y el mismo ``hash_texto`` (no recomputa).
"""

from __future__ import annotations

from src.embeddings.embedding_cache import EmbeddingCache


def test_todos_pendientes_si_no_hay_nada_en_cache():
    chunk_id_to_hash = {"c1": "hash1", "c2": "hash2"}
    pendientes = EmbeddingCache._pendientes_de(chunk_id_to_hash, {})
    assert set(pendientes) == {"c1", "c2"}


def test_chunk_ya_cacheado_con_mismo_hash_no_esta_pendiente():
    chunk_id_to_hash = {"c1": "hash1", "c2": "hash2"}
    cacheados = {"c1": "hash1"}
    pendientes = EmbeddingCache._pendientes_de(chunk_id_to_hash, cacheados)
    assert pendientes == ["c2"]


def test_chunk_con_hash_cambiado_vuelve_a_estar_pendiente():
    """Si el texto del chunk cambió (nuevo hash_texto), se debe recomputar."""
    chunk_id_to_hash = {"c1": "hash-nuevo"}
    cacheados = {"c1": "hash-viejo"}
    pendientes = EmbeddingCache._pendientes_de(chunk_id_to_hash, cacheados)
    assert pendientes == ["c1"]


def test_ningun_pendiente_si_todo_esta_cacheado():
    chunk_id_to_hash = {"c1": "hash1", "c2": "hash2"}
    cacheados = {"c1": "hash1", "c2": "hash2"}
    pendientes = EmbeddingCache._pendientes_de(chunk_id_to_hash, cacheados)
    assert pendientes == []
