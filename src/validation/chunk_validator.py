"""
ChunkValidator: validaciones duras y blandas antes de guardar en Mongo.

Validaciones duras (rechazo del fragmento):
- Campos obligatorios presentes y con el tipo correcto (Tabla 1).
- ``num_tokens`` dentro del límite del encoder (512 por defecto).
- ``posicion`` creciente y sin huecos dentro de un mismo ``doc_id``.
- ``chunk_id`` único dentro del documento.

Validaciones blandas (warning, se guarda con ``validation_warnings``):
- El texto termina en puntuación terminal (no corta una oración), salvo
  unidades atómicas justificadas (filas de tabla, elementos PBF).
- El texto empieza en límite de oración (no arranca a mitad).

Estas dos son un DETECTOR, no un corrector: cumplir el requisito de
completitud lingüística es responsabilidad de la estrategia de chunking
(``TextSegmenter.empaquetar_por_oraciones``). Lo que aquí se busca es que la
advertencia señale cortes reales y no ruido, para lo cual la heurística
distingue prosa de contenido estructural (encabezados, viñetas, filas de tabla,
pares ``clave: valor``) y tolera los cierres tipográficos habituales en PDF.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Set

from src.models.chunk import Chunk

logger = logging.getLogger(__name__)

# Campos obligatorios de la Tabla 1 y sus tipos esperados.
_CAMPOS_OBLIGATORIOS: dict = {
    "doc_id": str,
    "chunk_id": str,
    "fuente": str,
    "formato": str,
    "fenomeno": int,
    "posicion": int,
    "num_tokens": int,
    "texto": str,
}

# Puntuación que cierra una unidad de texto completa. Incluye puntos
# suspensivos y los dos puntos (encabezado de lista/enumeración), además de la
# puntuación terminal CJK y las comillas angulares de cierre.
_PUNTUACION_CIERRE = set(".!?…:»›。！？．｡")

# Signos que pueden ir DESPUÉS de la puntuación terminal: comillas y paréntesis
# de cierre. `«algo dicho.»` o `(ver nota.)` cierran oración igual que `algo.`
_RE_SIGNOS_CIERRE = re.compile(r"[\"'”’)\]\}\s]+$")

# Abreviaturas que nunca cierran una oración ("Art. 5", "Sr. Pérez", "vs.").
# Deliberadamente NO incluye abreviaturas ambiguas que también son palabras
# corrientes en español ("no", "al", "mar", "ha"...): terminar en "... que no."
# es un final de oración perfectamente válido.
_ABREVIATURAS_NO_FINALES = {
    "art", "arts", "cap", "dr", "dra", "ed", "eds", "fig", "figs", "núm",
    "num", "pág", "pag", "pp", "prof", "sr", "sra", "srta", "vol", "vs",
}

# Marcadores de lista/estructura al inicio de línea: viñetas, enumeraciones
# ("1.", "a)", "iv."), encabezados markdown, citas y filas de tabla.
_RE_MARCADOR_LISTA = re.compile(
    r"^\s*(?:[-–—•*·▪◦]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|#{1,6}|>|\|)(?:\s|$)"
)
# Par "clave: valor" (típico de tablas y fichas de datos extraídas de PDF).
_RE_CLAVE_VALOR = re.compile(r"^[^\n:]{1,40}:\s*\S")
# Línea que empieza a mitad de oración: minúscula o puntuación de continuación.
_RE_INICIO_CORTADO = re.compile(r"^[a-záéíóúñüàèìòùâêîôûãõäëïöç]|^[,;]")

# Palabras que jamás cierran una oración: si un texto corto termina así, es un
# corte, no un encabezado (artículos, preposiciones y conjunciones ES/EN).
_PALABRAS_NO_FINALES = {
    "a", "al", "ante", "bajo", "con", "contra", "de", "del", "desde", "durante",
    "e", "el", "en", "entre", "hacia", "hasta", "la", "las", "los", "mediante",
    "o", "para", "pero", "por", "que", "según", "si", "sin", "sobre", "su",
    "sus", "tras", "un", "una", "unos", "unas", "y", "and", "for", "from",
    "in", "of", "or", "the", "to", "with",
}

# Prefijos de numeración de encabezado: "3.", "3.1.2", "IV.", "Artículo 5".
_RE_NUMERACION_ENCABEZADO = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[.)]?\s|[IVXLC]+[.)]\s|"
    r"(?:art[íi]culo|cap[íi]tulo|secci[óo]n|anexo|tabla|figura|cuadro|gr[áa]fico|parte|t[íi]tulo)\b)",
    re.IGNORECASE,
)

# Longitud máxima (caracteres/palabras) para considerar una línea un encabezado.
# Son umbrales estrechos a propósito: ante la duda es preferible advertir de un
# corte que no existe a callar uno real.
_MAX_CARACTERES_ENCABEZADO = 80
_MAX_PALABRAS_ENCABEZADO = 6


@dataclass
class ValidationResult:
    """Resultado de la validación de un lote de fragmentos."""

    validos: List[Chunk] = field(default_factory=list)
    rechazados: List[Chunk] = field(default_factory=list)
    motivos: List[str] = field(default_factory=list)
    rechazos_detalle: List[dict] = field(default_factory=list)


class ChunkValidator:
    """Valida fragmentos; los que fallan validación dura se descartan."""

    def __init__(self, max_tokens: int = 512) -> None:
        self.max_tokens = max_tokens

    def validate(self, chunks: List[Chunk]) -> ValidationResult:
        """Valida el lote completo.

        Returns:
            :class:`ValidationResult` con los fragmentos válidos (con sus
            warnings) y los rechazados (con el motivo de cada rechazo).
        """
        resultado = ValidationResult()
        vistos: Set[str] = set()
        posicion_esperada = 0
        doc_id_actual: str | None = None

        for chunk in sorted(chunks, key=lambda c: (c.doc_id, c.posicion)):
            if doc_id_actual != chunk.doc_id:
                doc_id_actual = chunk.doc_id
                posicion_esperada = 0
            regla, motivo = self._validar_duro(chunk)
            if motivo is None:
                # Integridad de posiciones y unicidad de chunk_id por documento.
                if chunk.posicion != posicion_esperada:
                    regla, motivo = (
                        "posicion",
                        f"posicion {chunk.posicion} fuera de secuencia "
                        f"(se esperaba {posicion_esperada})",
                    )
                elif chunk.chunk_id in vistos:
                    regla, motivo = "duplicado", f"chunk_id duplicado: {chunk.chunk_id}"

            if motivo is not None:
                logger.warning(
                    "Chunk rechazado | doc=%s pos=%s fuente=%s | [%s] %s",
                    chunk.doc_id,
                    chunk.posicion,
                    chunk.fuente,
                    regla,
                    motivo,
                )
                resultado.rechazados.append(chunk)
                resultado.motivos.append(f"[{regla}] {motivo}")
                resultado.rechazos_detalle.append(
                    {
                        "doc_id": chunk.doc_id,
                        "chunk_id": chunk.chunk_id,
                        "posicion": chunk.posicion,
                        "fuente": chunk.fuente,
                        "regla": regla,
                        "motivo": motivo,
                        "texto": chunk.texto,
                    }
                )
                # Un rechazo por otra regla no debe romper la secuencia de los
                # fragmentos siguientes válidos: si este era el esperado, se
                # avanza la posición para evitar rechazos en cascada.
                if chunk.posicion == posicion_esperada:
                    posicion_esperada += 1
                continue

            vistos.add(chunk.chunk_id)
            posicion_esperada += 1

            # Validaciones blandas (warnings): no impiden el guardado.
            for advertencia in self._validar_blando(chunk):
                if advertencia not in chunk.validation_warnings:
                    chunk.validation_warnings.append(advertencia)
            resultado.validos.append(chunk)
        return resultado

    # Validaciones duras ----------------------------------------------------------

    def _validar_duro(self, chunk: Chunk) -> tuple[str, str | None]:
        """Devuelve ``(regla, motivo)``; ``motivo`` es ``None`` si pasa.

        ``regla`` identifica la regla de negocio incumplida (útil para logs
        y análisis de rechazos): ``obligatorio``, ``tipo``, ``fenomeno``,
        ``tokens``, ``posicion`` o ``duplicado``.
        """
        for campo, tipo in _CAMPOS_OBLIGATORIOS.items():
            valor = getattr(chunk, campo, None)
            if valor is None or (isinstance(valor, str) and not valor.strip()):
                return "obligatorio", f"campo obligatorio ausente: {campo}"
            if not isinstance(valor, tipo):
                return "tipo", f"campo '{campo}' con tipo incorrecto (esperado {tipo.__name__})"
        if chunk.fenomeno not in (1, 2, 3):
            return "fenomeno", f"fenomeno fuera de rango: {chunk.fenomeno}"
        if chunk.num_tokens > self.max_tokens:
            return (
                "tokens",
                f"num_tokens {chunk.num_tokens} supera el límite del encoder "
                f"({self.max_tokens})",
            )
        if chunk.posicion < 0:
            return "posicion", f"posicion negativa: {chunk.posicion}"
        return "ok", None

    # Validaciones blandas ----------------------------------------------------------

    def _validar_blando(self, chunk: Chunk) -> List[str]:
        """Advertencias que no impiden el guardado.

        Solo se advierte sobre PROSA: las unidades atómicas por formato
        (``csv``/``xlsx``/``pbf``) y las líneas estructurales de cualquier
        formato (encabezados, viñetas, filas de tabla, pares ``clave: valor``)
        no se juzgan con la regla de oración completa, porque no son oraciones.
        """
        advertencias: List[str] = []
        if chunk.formato in ("csv", "xlsx", "pbf"):
            return advertencias

        texto = chunk.texto.strip()
        if not texto:
            return advertencias

        lineas = [l.strip() for l in texto.splitlines() if l.strip()]
        primera_linea = lineas[0] if lineas else texto
        ultima_linea = lineas[-1] if lineas else texto

        if not _cierra_oracion(texto) and not _es_linea_estructural(ultima_linea):
            advertencias.append(
                "el texto no termina en puntuación terminal (posible oración cortada)"
            )

        if _RE_INICIO_CORTADO.match(primera_linea) and not _es_linea_estructural(primera_linea):
            advertencias.append("el texto parece empezar a mitad de una oración")
        return advertencias


# Heurísticas de completitud lingüística -------------------------------------------


def _cierra_oracion(texto: str) -> bool:
    """True si ``texto`` termina en una unidad de texto completa.

    Acepta la puntuación terminal seguida de comillas/paréntesis de cierre y
    descarta los puntos que en realidad pertenecen a una abreviatura o a una
    inicial ("Art.", "Sr.", "J."), que no cierran oración.
    """
    nucleo = _RE_SIGNOS_CIERRE.sub("", texto.rstrip())
    if not nucleo or nucleo[-1] not in _PUNTUACION_CIERRE:
        return False
    if nucleo[-1] != ".":
        return True
    ultima_palabra = re.split(r"[\s(«\"'“‘]", nucleo[:-1])[-1].lower()
    if len(ultima_palabra) == 1 and ultima_palabra.isalpha():
        return False  # inicial de un nombre propio ("J. Pérez")
    return ultima_palabra.strip(".") not in _ABREVIATURAS_NO_FINALES


def _es_linea_estructural(linea: str) -> bool:
    """True si la línea no es prosa y no debe juzgarse como oración.

    Cubre el contenido no oracional que dejan los extractores de PDF/HTML:
    viñetas y enumeraciones, encabezados, filas de tabla y pares
    ``clave: valor``.
    """
    if _RE_MARCADOR_LISTA.match(linea) or _RE_CLAVE_VALOR.match(linea):
        return True
    if linea.count("|") >= 2 or "\t" in linea:
        return True  # fila de tabla
    return _parece_encabezado(linea)


def _parece_encabezado(linea: str) -> bool:
    """True si la línea parece un título/encabezado (completo sin punto final).

    Un encabezado es corto, empieza en mayúscula o versales y no termina en una
    palabra que exija continuación (artículo, preposición o conjunción). La
    numeración de sección ("3.1", "Artículo 5") es la señal más fiable y relaja
    el límite de palabras; sin ella se exige brevedad, porque una línea de prosa
    cortada y un título breve son indistinguibles por su superficie.
    """
    palabras = linea.split()
    if not palabras or len(linea) > _MAX_CARACTERES_ENCABEZADO:
        return False
    if re.search(r"[.!?…]\s", linea):
        return False  # contiene una oración cerrada y sigue: es prosa cortada
    if linea.rstrip().endswith((".", "!", "?", "…")):
        # Un encabezado no lleva punto final: si llegó aquí terminando en punto
        # es porque el punto era de una abreviatura ("... en el Art."), y ahí sí
        # cabe advertir de que la oración puede continuar en el chunk siguiente.
        return False
    ultima = re.sub(r"[^\wáéíóúñü]+$", "", palabras[-1].lower())
    if ultima in _PALABRAS_NO_FINALES:
        return False  # "...de los" no es un título: es un corte
    if linea.isupper():
        return True
    if _RE_NUMERACION_ENCABEZADO.match(linea):
        return True
    if len(palabras) > _MAX_PALABRAS_ENCABEZADO:
        return False
    return palabras[0][:1].isupper()
