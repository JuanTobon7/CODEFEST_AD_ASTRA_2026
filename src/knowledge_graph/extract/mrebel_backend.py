"""Motor de inferencia mREBEL: carga lazy y generación de tripletas.

Aislado de la estrategia (:mod:`mrebel_relation_strategy`) para que el ciclo
de vida del modelo (descarga de HuggingFace, dispositivo CUDA/CPU, idioma de
origen del tokenizer mBART, beam search) sea reemplazable y testeable con
fakes sin tocar la lógica de pares de entidades.

``tokenizer``/``modelo`` inyectables (tests u otro checkpoint); si faltan,
se cargan de HuggingFace en el PRIMER uso (no al instanciar).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from src.knowledge_graph.extract.mrebel_config import SRC_LANG_DEFAULT, TGT_LANG

logger = logging.getLogger(__name__)


class MrebelInferenceEngine:
    """Genera tripletas ``<triplet> sujeto <subj> relación <obj> objeto``.

    Args:
        model_id: checkpoint de HuggingFace (mREBEL, seq2seq multilingüe).
        src_lang: idioma de origen del tokenizer mBART (p. ej. ``en_XX``).
        device_preference: ``None``/``"auto"`` (detecta CUDA), ``"cpu"`` o ``"cuda"``.
        max_length: longitud máxima de la secuencia generada.
        num_beams: beam search (3 = el de la model card).
        use_fp16: si ``True`` y hay CUDA, convierte el modelo a float16
            (VRAM ~2.9 GB → ~1.5 GB y ~2x más rápido en RTX 30xx); en CPU
            se ignora con warning (FP16 en CPU es más lento).
        tokenizer/modelo: inyectables (tests u otro checkpoint).
    """

    def __init__(
        self,
        model_id: str,
        src_lang: str = SRC_LANG_DEFAULT,
        device_preference: Optional[str] = None,
        max_length: int = 256,
        num_beams: int = 3,
        use_fp16: bool = False,
        tokenizer=None,
        modelo=None,
    ) -> None:
        self._model_id = model_id
        self._src_lang = src_lang
        self._device_preference = device_preference
        self._max_length = max_length
        self._num_beams = num_beams
        self._use_fp16 = use_fp16
        self._tokenizer = tokenizer
        self._modelo = modelo
        self._cargado = False
        self._device: str = "cpu"

    # -- Ciclo de vida del modelo -------------------------------------------

    def asegurar_modelo(self) -> None:
        """Carga tokenizer/modelo (lazy) y los mueve al dispositivo."""
        if self._cargado:
            return
        self._device = self.resolver_device()
        if self._tokenizer is None or self._modelo is None:
            try:
                from transformers import (  # import local: solo si se usa mREBEL
                    AutoModelForSeq2SeqLM,
                    AutoTokenizer,
                )
            except ImportError as exc:  # pragma: no cover - entorno sin transformers
                raise ImportError(
                    "mrebel requiere 'transformers' instalado "
                    "(pip install transformers). El flujo por defecto "
                    "(coocurrencia-oracional) no necesita este paquete."
                ) from exc
            if self._tokenizer is None:
                logger.info("Descargando tokenizer '%s'...", self._model_id)
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_id, src_lang=self._src_lang, tgt_lang=TGT_LANG
                )
            if self._modelo is None:
                logger.info("Descargando modelo mREBEL '%s'...", self._model_id)
                self._modelo = AutoModelForSeq2SeqLM.from_pretrained(self._model_id)
        if hasattr(self._modelo, "to"):
            self._modelo.to(self._device)
        if self._use_fp16 and self._device == "cuda" and hasattr(self._modelo, "half"):
            self._modelo.half()
        elif self._use_fp16:
            logger.warning(
                "use_fp16=True requiere CUDA; se continúa en FP32 (device=%s)",
                self._device,
            )
        if hasattr(self._modelo, "eval"):
            self._modelo.eval()
        self._cargado = True
        logger.info(
            "Modelo mREBEL listo en '%s'%s",
            self._device,
            " (fp16)" if self._use_fp16 and self._device == "cuda" else " (fp32)",
        )

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

    # -- Generación ---------------------------------------------------------

    def generar(self, texto: str) -> List[str]:
        """Genera las secuencias decodificadas de tripletas para ``texto``.

        Sigue el flujo de la model card de mREBEL: tokeniza el texto,
        genera con ``decoder_start_token_id=tp_XX`` y decodifica SIN omitir
        los tokens especiales (necesarios para parsear las tripletas).
        """
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("mrebel requiere 'torch' instalado.") from exc

        entradas = self._tokenizer(
            texto,
            max_length=self._max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        ids = entradas["input_ids"].to(self._device)
        mascara = entradas["attention_mask"].to(self._device)
        decoder_start = self._tokenizer.convert_tokens_to_ids(TGT_LANG)
        with torch.no_grad():
            generados = self._modelo.generate(
                ids,
                attention_mask=mascara,
                decoder_start_token_id=decoder_start,
                max_length=self._max_length,
                num_beams=self._num_beams,
                num_return_sequences=self._num_beams,
            )
        return self._tokenizer.batch_decode(generados, skip_special_tokens=False)