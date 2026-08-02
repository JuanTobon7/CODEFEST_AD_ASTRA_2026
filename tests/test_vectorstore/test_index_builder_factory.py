"""
Tests: ``IndexBuilderFactory`` — registro por decorador y resolución de
``FAISS_INDEX_TYPE=auto`` según el umbral configurado.
"""

from __future__ import annotations

import pytest

# Registran flat_ip/ivf_flat/hnsw vía @IndexBuilderFactory.register.
from src.vectorstore import flat_ip_strategy, hnsw_strategy, ivf_flat_strategy  # noqa: F401
from src.vectorstore.index_builder_base import IndexBuildConfig
from src.vectorstore.index_builder_factory import IndexBuilderFactory


def test_encoders_por_defecto_estan_registrados():
    assert {"flat_ip", "ivf_flat", "hnsw"}.issubset(set(IndexBuilderFactory.list_available()))


def test_create_encoder_desconocido_lanza_value_error():
    with pytest.raises(ValueError):
        IndexBuilderFactory.create("no-existe")


def test_create_flat_ip_construye_indice_exacto():
    estrategia = IndexBuilderFactory.create("flat_ip")
    config = IndexBuildConfig()
    indice = estrategia.build(dim=8, config=config)
    assert indice.d == 8
    assert estrategia.requires_training is False


def test_resolve_auto_elige_flat_ip_bajo_el_umbral():
    config = IndexBuildConfig(ivf_auto_threshold=1000)
    estrategia = IndexBuilderFactory.resolve("auto", n_vectors=10, config=config)
    assert estrategia.index_type_name == "flat_ip"


def test_resolve_auto_elige_ivf_flat_sobre_el_umbral():
    config = IndexBuildConfig(ivf_auto_threshold=1000)
    estrategia = IndexBuilderFactory.resolve("auto", n_vectors=5000, config=config)
    assert estrategia.index_type_name == "ivf_flat"


def test_resolve_tipo_explicito_ignora_el_umbral():
    config = IndexBuildConfig(ivf_auto_threshold=1)
    estrategia = IndexBuilderFactory.resolve("hnsw", n_vectors=1, config=config)
    assert estrategia.index_type_name == "hnsw"
