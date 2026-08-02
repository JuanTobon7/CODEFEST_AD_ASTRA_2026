"""
Tests: ``EncoderFactory`` debe validar la licencia contra la lista blanca
antes de instanciar, y rechazar en el registro los encoders que no cubran
es/en/pt salvo que se marquen como complementarios (Sección 6).
"""

from __future__ import annotations

import pytest

from src.encoders.base import EncoderConfig, EncoderStrategy, LicenseNotAllowedError
from src.encoders.factory import EncoderFactory


class _FakeGPLEncoder(EncoderStrategy):
    """Encoder ficticio con licencia no permitida (fixture de test)."""

    model_id = "fake/gpl-encoder"
    embedding_dim = 8
    max_input_tokens = 512
    supported_languages = ["es", "en", "pt"]
    license = "gpl-3.0"

    def _cargar_modelo(self, device: str):
        return object()


class _FakeAllowedEncoder(EncoderStrategy):
    """Encoder ficticio con licencia MIT (permitida)."""

    model_id = "fake/mit-encoder"
    embedding_dim = 8
    max_input_tokens = 512
    supported_languages = ["es", "en", "pt"]
    license = "mit"

    def _cargar_modelo(self, device: str):
        return object()


class _FakeMonolingualEncoder(EncoderStrategy):
    """Encoder ficticio que no cubre es/en/pt (prueba el rechazo en registro)."""

    model_id = "fake/pt-only"
    embedding_dim = 8
    max_input_tokens = 512
    supported_languages = ["pt"]
    license = "mit"

    def _cargar_modelo(self, device: str):
        return object()


@pytest.fixture(autouse=True)
def _registro_aislado():
    """Aísla el registro global del Factory entre tests (no afecta encoders reales)."""
    respaldo = dict(EncoderFactory._registry)
    yield
    EncoderFactory._registry.clear()
    EncoderFactory._registry.update(respaldo)


def test_licencia_no_permitida_rechaza_instanciacion():
    EncoderFactory.register("fake-gpl")(_FakeGPLEncoder)
    with pytest.raises(LicenseNotAllowedError):
        EncoderFactory.create("fake-gpl")


def test_licencia_no_permitida_se_puede_forzar_conscientemente():
    EncoderFactory.register("fake-gpl-2")(_FakeGPLEncoder)
    instancia = EncoderFactory.create("fake-gpl-2", EncoderConfig(allow_unlisted_license=True))
    assert isinstance(instancia, _FakeGPLEncoder)


def test_licencia_permitida_instancia_correctamente():
    EncoderFactory.register("fake-mit")(_FakeAllowedEncoder)
    instancia = EncoderFactory.create("fake-mit")
    assert instancia.license == "mit"


def test_encoder_desconocido_lanza_value_error():
    with pytest.raises(ValueError):
        EncoderFactory.create("no-existe-este-encoder")


def test_registro_rechaza_idiomas_incompletos_sin_marca_complementaria():
    with pytest.raises(ValueError):
        EncoderFactory.register("fake-pt-only")(_FakeMonolingualEncoder)


def test_registro_permite_idiomas_incompletos_si_es_complementario():
    _FakeMonolingualEncoder.is_complementary = True
    try:
        EncoderFactory.register("fake-pt-only-ok")(_FakeMonolingualEncoder)
        assert "fake-pt-only-ok" in EncoderFactory._registry
    finally:
        _FakeMonolingualEncoder.is_complementary = False


def test_list_available_expone_metadata_sin_instanciar():
    EncoderFactory.register("fake-mit-listado")(_FakeAllowedEncoder)
    disponibles = {d["name"]: d for d in EncoderFactory.list_available()}
    assert "fake-mit-listado" in disponibles
    assert disponibles["fake-mit-listado"]["license"] == "mit"
    assert disponibles["fake-mit-listado"]["embedding_dim"] == 8
