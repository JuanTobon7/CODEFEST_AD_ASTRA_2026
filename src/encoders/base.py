"""
Estrategia abstracta de codificación semántica (patrón Strategy, Sección 4.2/4.3).

Cada implementación concreta envuelve un checkpoint **BERT** real de
HuggingFace (``google-bert/*``, BETO, BERTimbau, ``prajjwal1/bert-tiny``...)
— nunca un decoder (GPT/LLaMA/Gemini/Claude) — ensamblado vía
``sentence_transformers.models.Transformer`` (que carga el modelo con
``transformers.AutoModel`` internamente) + *mean pooling*, y autodeclara los
6 criterios de selección de la especificación: soporte multilingüe,
dimensionalidad, longitud máxima de entrada, benchmark MTEB/BEIR, licencia y
eficiencia.
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Reglas de negocio centralizadas (Sección 6): cada EncoderStrategy es el
# Information Expert que las evalúa sobre sus propios atributos declarados.
LICENSE_WHITELIST = {"apache-2.0", "mit", "cc-by-4.0"}
IDIOMAS_MINIMOS = {"es", "en", "pt"}

_APROX_CHARS_POR_TOKEN = 4
_REGEX_ORACIONES = re.compile(r"(?<=[.!?。！？])\s+")


class LicenseNotAllowedError(RuntimeError):
    """La licencia del modelo no está en la lista blanca permitida."""


class EncoderConfig(BaseModel):
    """Configuración runtime inyectada a cada :class:`EncoderStrategy`."""

    batch_size: int = Field(default=32, ge=1)
    device_preference: Optional[Literal["cpu", "cuda", "mps"]] = None
    allow_unlisted_license: bool = False


class EncoderStrategy(ABC):
    """Interfaz común de codificación semántica intercambiable (Strategy)."""

    # Metadata declarativa de los 6 criterios (definida por cada subclase).
    model_id: str
    embedding_dim: int
    max_input_tokens: int
    supported_languages: List[str]
    license: str
    mteb_retrieval_score: Optional[float] = None
    benchmark_reference: Optional[str] = None
    requires_prefix: Optional[Dict[str, str]] = None
    is_complementary: bool = False
    device_preference: Literal["cpu", "cuda", "mps"] = "cpu"

    def __init__(self, config: Optional[EncoderConfig] = None) -> None:
        self.config = config or EncoderConfig()
        self._model = None
        self.avg_encode_time_ms_per_batch: Optional[float] = None
        if self.config.device_preference:
            self.device_preference = self.config.device_preference

    @property
    def name(self) -> str:
        """Nombre corto registrado en el :class:`EncoderFactory`."""
        return getattr(self, "_registry_name", self.model_id)

    # -- Information Expert: reglas de negocio sobre la propia metadata ----

    @classmethod
    def cubre_idiomas_minimos(cls) -> bool:
        """True si ``supported_languages`` cubre es/en/pt, o si es complementario.

        Solo la estrategia conoce sus propios idiomas soportados: es la
        responsable de decidir si cumple la regla (Sección 6).
        """
        idiomas = {i.lower() for i in cls.supported_languages}
        return IDIOMAS_MINIMOS.issubset(idiomas) or cls.is_complementary

    @classmethod
    def licencia_permitida(cls, allow_unlisted: bool = False) -> bool:
        """True si ``license`` está en la lista blanca, o se fuerza explícitamente.

        Solo la estrategia conoce su propia licencia: es la responsable de
        decidir si cumple la regla (Sección 6), no quien la instancia.
        """
        return allow_unlisted or cls.license.lower() in LICENSE_WHITELIST

    def load(self) -> None:
        """Carga perezosa del modelo: nunca se invoca desde ``__init__``."""
        if self._model is not None:
            return
        device = self._resolver_dispositivo()
        logger.info("Cargando encoder '%s' (%s) en %s", self.model_id, self.__class__.__name__, device)
        self._model = self._cargar_modelo(device)
        self.device_preference = device

    @abstractmethod
    def _cargar_modelo(self, device: str):
        """Instancia el modelo subyacente (``sentence-transformers``)."""

    def _resolver_dispositivo(self) -> str:
        """``cuda`` > ``mps`` > ``cpu``, salvo que ``device_preference`` fuerce uno."""
        preferencia = self.config.device_preference
        if preferencia and preferencia != "auto":
            return preferencia
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:  # pragma: no cover - torch no disponible
            pass
        return "cpu"

    def _prefijo(self, is_query: bool) -> str:
        """Prefijo query/passage si el modelo lo requiere (p. ej. familia E5)."""
        if not self.requires_prefix:
            return ""
        clave = "query" if is_query else "passage"
        return self.requires_prefix.get(clave, "")

    def contar_tokens(self, texto: str) -> int:
        """Cuenta tokens con el tokenizador propio del modelo (no el genérico
        usado en el chunking, que puede diferir). Solo la estrategia tiene
        acceso al tokenizador real de su modelo una vez cargado.
        """
        self.load()
        tokenizador = getattr(self._model, "tokenizer", None)
        if tokenizador is not None:
            try:
                return len(tokenizador.encode(texto, add_special_tokens=True))
            except Exception:  # pragma: no cover - tokenizador incompatible
                pass
        return max(1, len(texto) // _APROX_CHARS_POR_TOKEN)

    def excede_limite(self, texto: str) -> bool:
        """True si ``texto`` supera ``max_input_tokens`` de este encoder."""
        return self.contar_tokens(texto) > self.max_input_tokens

    def ajustar_a_limite(self, texto: str) -> Optional[str]:
        """Trunca ``texto`` por oraciones completas hasta caber en el límite.

        Devuelve el texto sin cambios si ya cabe, el texto truncado si es
        posible preservar al menos una oración, o ``None`` si ni la primera
        oración cabe (el llamador debe excluir el chunk para ese encoder).
        """
        if not self.excede_limite(texto):
            return texto
        oraciones = _REGEX_ORACIONES.split(texto.strip())
        if not oraciones or not oraciones[0]:
            return None
        acumulado = ""
        for oracion in oraciones:
            candidato = f"{acumulado} {oracion}".strip()
            if self.contar_tokens(candidato) > self.max_input_tokens:
                break
            acumulado = candidato
        return acumulado or None

    def encode(self, texts: List[str], is_query: bool = False, batch_size: Optional[int] = None) -> np.ndarray:
        """Codifica ``texts`` en vectores normalizados a norma unitaria.

        El prefijo query/passage (si aplica) se resuelve aquí dentro, nunca
        en el orquestador (encapsulamiento del detalle interno del modelo).

        Raises:
            AssertionError: si la dimensión real difiere de ``embedding_dim``.
        """
        self.load()
        prefijo = self._prefijo(is_query)
        entradas = [f"{prefijo}{t}" for t in texts] if prefijo else list(texts)
        tam_lote = batch_size or self.config.batch_size

        inicio = time.perf_counter()
        vectores = self._model.encode(  # type: ignore[union-attr]
            entradas,
            batch_size=tam_lote,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        duracion_ms = (time.perf_counter() - inicio) * 1000
        n_lotes = max(1, -(-len(entradas) // tam_lote))
        self.avg_encode_time_ms_per_batch = duracion_ms / n_lotes

        vectores = np.asarray(vectores, dtype=np.float32)
        if vectores.ndim == 1:
            vectores = vectores.reshape(1, -1)
        assert vectores.shape[1] == self.embedding_dim, (
            f"{self.__class__.__name__}: dimensión real {vectores.shape[1]} "
            f"!= embedding_dim declarado {self.embedding_dim}"
        )
        return vectores

    def to_metadata(self) -> Dict[str, object]:
        """Serializa los 6 criterios de selección para trazabilidad."""
        return {
            "encoder_name": self.name,
            "model_id": self.model_id,
            "embedding_dim": self.embedding_dim,
            "max_input_tokens": self.max_input_tokens,
            "supported_languages": list(self.supported_languages),
            "license": self.license,
            "mteb_retrieval_score": self.mteb_retrieval_score,
            "benchmark_reference": self.benchmark_reference,
            "avg_encode_time_ms_per_batch": self.avg_encode_time_ms_per_batch,
            "device_preference": self.device_preference,
            "is_complementary": self.is_complementary,
        }
