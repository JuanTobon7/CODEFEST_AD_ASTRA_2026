"""Motor de inferencia NLI: carga lazy, dispositivo y clasificación.

Aislado de la estrategia (:mod:`nli_relation_strategy`) para que el ciclo
de vida del modelo (descarga de HuggingFace, dispositivo CUDA/CPU, índice
de la etiqueta entailment, softmax) sea reemplazable y testeable con fakes
sin tocar la lógica de pares de entidades.

``tokenizer``/``modelo`` inyectables (tests u otro checkpoint); si faltan,
se cargan de HuggingFace en el PRIMER uso (no al instanciar).
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class NLIInferenceEngine:
    """Clasificador NLI de un lote de (premisa, hipótesis) -> entailment.

    Args:
        model_id: checkpoint de HuggingFace (clasificación NLI multilingüe).
        device_preference: ``None``/``"auto"`` (detecta CUDA), ``"cpu"`` o ``"cuda"``.
        max_length: truncado de la secuencia premisa+hipótesis.
        batch_size: tamaño de lote del forward (particiona las hipótesis).
        use_fp16: si hay CUDA, corre el modelo en media precisión (~2x más
            rápido y la mitad de VRAM); en CPU se ignora.
        tokenizer/modelo: inyectables (tests u otro checkpoint).
    """

    def __init__(
        self,
        model_id: str,
        device_preference: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = 16,
        use_fp16: bool = False,
        tokenizer=None,
        modelo=None,
    ) -> None:
        self._model_id = model_id
        self._device_preference = device_preference
        self._max_length = max_length
        self._batch_size = batch_size
        self._use_fp16 = use_fp16
        self._tokenizer = tokenizer
        self._modelo = modelo
        self._cargado = False
        self._entailment_idx: int = 2
        self._device: str = "cpu"

    # -- Ciclo de vida del modelo -------------------------------------------

    def asegurar_modelo(self) -> None:
        """Carga tokenizer/modelo (lazy) y deriva el índice de entailment."""
        if self._cargado:
            return
        if self._tokenizer is None or self._modelo is None:
            try:
                from transformers import (  # import local: solo si se usa NLI
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
            except ImportError as exc:  # pragma: no cover - entorno sin transformers
                raise ImportError(
                    "nli-zero-shot requiere 'transformers' instalado "
                    "(pip install transformers). El flujo por defecto "
                    "(coocurrencia-oracional) no necesita este paquete."
                ) from exc
            if self._tokenizer is None:
                logger.info("Descargando tokenizer '%s'...", self._model_id)
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            if self._modelo is None:
                logger.info("Descargando modelo NLI '%s'...", self._model_id)
                self._modelo = AutoModelForSequenceClassification.from_pretrained(
                    self._model_id
                )
        self._entailment_idx = self.indice_entailment(self._modelo)
        self._device = self.resolver_device()
        if hasattr(self._modelo, "to"):
            self._modelo.to(self._device)
        if self._use_fp16 and self._device.startswith("cuda") and hasattr(self._modelo, "half"):
            self._modelo.half()
        if hasattr(self._modelo, "eval"):
            self._modelo.eval()
        self._cargado = True
        logger.info(
            "Modelo NLI listo en '%s' (entailment_idx=%d, fp16=%s)",
            self._device,
            self._entailment_idx,
            self._use_fp16 and self._device.startswith("cuda"),
        )

    @staticmethod
    def indice_entailment(modelo) -> int:
        """Índice de la etiqueta 'entailment' según ``config.id2label``.

        No todos los checkpoints MNLI ordenan las 3 etiquetas igual; se
        deriva del config para no asumir el orden. Fallback conservador: 2.
        """
        cfg = getattr(modelo, "config", None)
        id2label = getattr(cfg, "id2label", None)
        if isinstance(id2label, dict):
            for indice, etiqueta in id2label.items():
                if "entail" in str(etiqueta).lower():
                    return int(indice)
        return 2

    def resolver_device(self) -> str:
        """Dispositivo de inferencia: preferencia explícita o auto (CUDA/CPU)."""
        pref = self._device_preference
        if pref is None or pref == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:  # pragma: no cover
                return "cpu"
        return pref

    # -- Clasificación -------------------------------------------------------

    def puntuar(self, premisas: List[str], hipotesis: List[str]) -> List[float]:
        """Probabilidad de entailment de cada (premisa, hipótesis) en batch."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("nli-zero-shot requiere 'torch' instalado.") from exc

        puntajes: List[float] = []
        for inicio in range(0, len(premisas), self._batch_size):
            bloque_p = premisas[inicio : inicio + self._batch_size]
            bloque_h = hipotesis[inicio : inicio + self._batch_size]
            entradas = self._tokenizer(
                bloque_p,
                bloque_h,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            entradas = {k: v.to(self._device) for k, v in entradas.items()}
            with torch.no_grad():
                logits = getattr(self._modelo(**entradas), "logits")
            probs = torch.softmax(logits.float(), dim=-1)
            puntajes.extend(probs[:, self._entailment_idx].tolist())
        return puntajes
