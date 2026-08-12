"""Estrategia NER simbólica (regex + gazetteer), sin modelos generativos.

Requisito duro del reto: en indexación y recuperación NO se permite invocar
LLMs/decoders. Esta estrategia reconoce entidades nombradas con:

1. Un gazetteer del dominio (organizaciones, países, programas espaciales,
   tecnologías IA, conceptos de seguridad espacial/territorial) tipado por
   :class:`EntityType`.
2. Matching por spans: las entidades de más palabras ganan sobre las
   subcadenas (p. ej. "inteligencia artificial" antes que "inteligencia").
3. Patrón de nombres propios multi-palabra (opcional) para entidades no
   catalogadas (``EntityType.OTRO``).

Es 100% determinista, offline y testeable; implementa :class:`EntityExtractor`
(Strategy/OCP: un modelo NER neuronal futuro solo necesita implementar la
misma interfaz).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Pattern, Tuple

from src.knowledge_graph.extract.base import EntityExtractor, normalizar_id_entidad
from src.knowledge_graph.extract.factory import EntityExtractorFactory
from src.knowledge_graph.models import Entity, EntityType

logger = logging.getLogger(__name__)

# -- Gazetteer del dominio (es/en, relevante a F1/F2/F3 del corpus) --------
# Clave: forma léxica en minúsculas; valor: tipo de entidad. Se normaliza
# internamente (sin acentos) para el matching.
GAZETTEER: Dict[str, EntityType] = {
    # Organizaciones
    "nasa": EntityType.ORGANIZACION,
    "esa": EntityType.ORGANIZACION,
    "aem": EntityType.ORGANIZACION,
    "jaxa": EntityType.ORGANIZACION,
    "isro": EntityType.ORGANIZACION,
    "cnsa": EntityType.ORGANIZACION,
    "roskosmos": EntityType.ORGANIZACION,
    "conae": EntityType.ORGANIZACION,
    "cenia": EntityType.ORGANIZACION,
    "agencias espaciales": EntityType.ORGANIZACION,
    "agencia espacial mexicana": EntityType.ORGANIZACION,
    "agencia espacial brasileña": EntityType.ORGANIZACION,
    "agencia espacial europea": EntityType.ORGANIZACION,
    "aeronáutica y del espacio": EntityType.ORGANIZACION,
    "administración nacional de aeronáutica": EntityType.ORGANIZACION,
    "spacex": EntityType.ORGANIZACION,
    "blue origin": EntityType.ORGANIZACION,
    "boeing": EntityType.ORGANIZACION,
    "lockheed martin": EntityType.ORGANIZACION,
    "darpa": EntityType.ORGANIZACION,
    "comisión nacional de actividades espaciales": EntityType.ORGANIZACION,
    # Organizaciones internacionales
    "onu": EntityType.ORGANIZACION_INTERNACIONAL,
    "organización de las naciones unidas": EntityType.ORGANIZACION_INTERNACIONAL,
    "united nations": EntityType.ORGANIZACION_INTERNACIONAL,
    "unoosa": EntityType.ORGANIZACION_INTERNACIONAL,
    "oficina de asuntos del espacio ultraterrestre": EntityType.ORGANIZACION_INTERNACIONAL,
    "uit": EntityType.ORGANIZACION_INTERNACIONAL,
    "unión internacional de telecomunicaciones": EntityType.ORGANIZACION_INTERNACIONAL,
    "cospar": EntityType.ORGANIZACION_INTERNACIONAL,
    "oea": EntityType.ORGANIZACION_INTERNACIONAL,
    "organización de los estados americanos": EntityType.ORGANIZACION_INTERNACIONAL,
    "celac": EntityType.ORGANIZACION_INTERNACIONAL,
    "unión europea": EntityType.ORGANIZACION_INTERNACIONAL,
    "otan": EntityType.ORGANIZACION_INTERNACIONAL,
    "unesco": EntityType.ORGANIZACION_INTERNACIONAL,
    "comité para usos pacíficos del espacio": EntityType.ORGANIZACION_INTERNACIONAL,
    # Países y regiones
    "méxico": EntityType.PAIS,
    "mexico": EntityType.PAIS,
    "estados unidos": EntityType.PAIS,
    "ee. uu.": EntityType.PAIS,
    "eeuu": EntityType.PAIS,
    "ee.uu.": EntityType.PAIS,
    "china": EntityType.PAIS,
    "rusia": EntityType.PAIS,
    "brasil": EntityType.PAIS,
    "argentina": EntityType.PAIS,
    "chile": EntityType.PAIS,
    "colombia": EntityType.PAIS,
    "perú": EntityType.PAIS,
    "peru": EntityType.PAIS,
    "ecuador": EntityType.PAIS,
    "bolivia": EntityType.PAIS,
    "venezuela": EntityType.PAIS,
    "paraguay": EntityType.PAIS,
    "uruguay": EntityType.PAIS,
    "canadá": EntityType.PAIS,
    "canada": EntityType.PAIS,
    "japón": EntityType.PAIS,
    "japon": EntityType.PAIS,
    "india": EntityType.PAIS,
    "francia": EntityType.PAIS,
    "alemania": EntityType.PAIS,
    "reino unido": EntityType.PAIS,
    "españa": EntityType.PAIS,
    "espana": EntityType.PAIS,
    "portugal": EntityType.PAIS,
    "corea del sur": EntityType.PAIS,
    "emiratos árabes unidos": EntityType.PAIS,
    "israel": EntityType.PAIS,
    "ucrania": EntityType.PAIS,
    "américa latina": EntityType.LUGAR,
    "latinoamérica": EntityType.LUGAR,
    "centroamérica": EntityType.LUGAR,
    "el caribe": EntityType.LUGAR,
    "antártida": EntityType.LUGAR,
    "antartida": EntityType.LUGAR,
    "órbita terrestre baja": EntityType.LUGAR,
    "órbita geoestacionaria": EntityType.LUGAR,
    "espacio ultraterrestre": EntityType.LUGAR,
    # Programas / misiones / sistemas
    "iss": EntityType.PROGRAMA,
    "estación espacial internacional": EntityType.PROGRAMA,
    "programa artemisa": EntityType.PROGRAMA,
    "artemisa": EntityType.PROGRAMA,
    "gateway": EntityType.PROGRAMA,
    "colmena": EntityType.PROGRAMA,
    "starlink": EntityType.PROGRAMA,
    "oneweb": EntityType.PROGRAMA,
    "gps": EntityType.PROGRAMA,
    "galileo": EntityType.PROGRAMA,
    "glonass": EntityType.PROGRAMA,
    "beidou": EntityType.PROGRAMA,
    "cbers": EntityType.PROGRAMA,
    "peacesat": EntityType.PROGRAMA,
    "constelación de satélites": EntityType.PROGRAMA,
    # Tecnologías / conceptos IA
    "inteligencia artificial": EntityType.TECNOLOGIA,
    "ia": EntityType.TECNOLOGIA,
    "machine learning": EntityType.TECNOLOGIA,
    "aprendizaje automático": EntityType.TECNOLOGIA,
    "deep learning": EntityType.TECNOLOGIA,
    "aprendizaje profundo": EntityType.TECNOLOGIA,
    "procesamiento de lenguaje natural": EntityType.TECNOLOGIA,
    "visión por computador": EntityType.TECNOLOGIA,
    "visión por computadora": EntityType.TECNOLOGIA,
    "internet de las cosas": EntityType.TECNOLOGIA,
    "iot": EntityType.TECNOLOGIA,
    "big data": EntityType.TECNOLOGIA,
    "blockchain": EntityType.TECNOLOGIA,
    "computación en la nube": EntityType.TECNOLOGIA,
    "computación cuántica": EntityType.TECNOLOGIA,
    "gemelos digitales": EntityType.TECNOLOGIA,
    "gemelo digital": EntityType.TECNOLOGIA,
    "modelos de lenguaje": EntityType.TECNOLOGIA,
    "modelos fundacionales": EntityType.TECNOLOGIA,
    "chatbot": EntityType.TECNOLOGIA,
    "asistentes virtuales": EntityType.TECNOLOGIA,
    "satélite": EntityType.TECNOLOGIA,
    "satélites": EntityType.TECNOLOGIA,
    "nanosatélite": EntityType.TECNOLOGIA,
    "cubesat": EntityType.TECNOLOGIA,
    "cubesats": EntityType.TECNOLOGIA,
    "teleobservación": EntityType.TECNOLOGIA,
    "teledetección": EntityType.TECNOLOGIA,
    "radar de apertura sintética": EntityType.TECNOLOGIA,
    "lidar": EntityType.TECNOLOGIA,
    "vehículos aéreos no tripulados": EntityType.TECNOLOGIA,
    "drones": EntityType.TECNOLOGIA,
    "propulsión eléctrica": EntityType.TECNOLOGIA,
    "cohetes reutilizables": EntityType.TECNOLOGIA,
    # Seguridad espacial / territorial
    "basura espacial": EntityType.CONFLICTO,
    "desechos orbitales": EntityType.CONFLICTO,
    "desechos espaciales": EntityType.CONFLICTO,
    "ciberseguridad": EntityType.SECTOR,
    "ciberataques": EntityType.CONFLICTO,
    "ciberguerra": EntityType.CONFLICTO,
    "guerra electrónica": EntityType.CONFLICTO,
    "guerra híbrida": EntityType.CONFLICTO,
    "guerra de información": EntityType.CONFLICTO,
    "interferencia de señales": EntityType.CONFLICTO,
    "jamming": EntityType.CONFLICTO,
    "spoofing": EntityType.CONFLICTO,
    "bloqueo de señales": EntityType.CONFLICTO,
    "ataques cibernéticos": EntityType.CONFLICTO,
    "seguridad nacional": EntityType.SECTOR,
    "seguridad espacial": EntityType.SECTOR,
    "seguridad territorial": EntityType.SECTOR,
    "soberanía": EntityType.SECTOR,
    "soberanía digital": EntityType.SECTOR,
    "soberanía espacial": EntityType.SECTOR,
    "defensa": EntityType.SECTOR,
    "inteligencia militar": EntityType.SECTOR,
    "vigilancia satelital": EntityType.TECNOLOGIA,
    "reconocimiento satelital": EntityType.TECNOLOGIA,
    # Eventos / tratados
    "tratado del espacio ultraterrestre": EntityType.EVENTO,
    "tratado de no proliferación": EntityType.EVENTO,
    "acuerdo artemisa": EntityType.EVENTO,
    "acuerdos artemisa": EntityType.EVENTO,
    "día internacional del espacio": EntityType.EVENTO,
    "pandemia": EntityType.EVENTO,
    # Sectores
    "agricultura": EntityType.SECTOR,
    "minería": EntityType.SECTOR,
    "energía": EntityType.SECTOR,
    "telecomunicaciones": EntityType.SECTOR,
    "transporte": EntityType.SECTOR,
    "salud": EntityType.SECTOR,
    "educación": EntityType.SECTOR,
    "ganadería": EntityType.SECTOR,
    "pesca": EntityType.SECTOR,
    "gestión de riesgos": EntityType.SECTOR,
    "protección civil": EntityType.SECTOR,
    "defensa civil": EntityType.SECTOR,
    # --- Formas en inglés (el corpus es mayormente EN) ---
    # Organizaciones
    "mexican space agency": EntityType.ORGANIZACION,
    "brazilian space agency": EntityType.ORGANIZACION,
    "european space agency": EntityType.ORGANIZACION,
    "national aeronautics and space administration": EntityType.ORGANIZACION,
    "space agencies": EntityType.ORGANIZACION,
    # Organizaciones internacionales
    "european union": EntityType.ORGANIZACION_INTERNACIONAL,
    "nato": EntityType.ORGANIZACION_INTERNACIONAL,
    "oas": EntityType.ORGANIZACION_INTERNACIONAL,
    "itu": EntityType.ORGANIZACION_INTERNACIONAL,
    # Países / regiones
    "russia": EntityType.PAIS,
    "united kingdom": EntityType.PAIS,
    "spain": EntityType.PAIS,
    "japan": EntityType.PAIS,
    "france": EntityType.PAIS,
    "germany": EntityType.PAIS,
    "canada": EntityType.PAIS,
    "south korea": EntityType.PAIS,
    "united arab emirates": EntityType.PAIS,
    "israel": EntityType.PAIS,
    "ukraine": EntityType.PAIS,
    "united states": EntityType.PAIS,
    "usa": EntityType.PAIS,
    "latin america": EntityType.LUGAR,
    "central america": EntityType.LUGAR,
    "the caribbean": EntityType.LUGAR,
    "antarctica": EntityType.LUGAR,
    "low earth orbit": EntityType.LUGAR,
    "leo": EntityType.LUGAR,
    "geostationary orbit": EntityType.LUGAR,
    "outer space": EntityType.LUGAR,
    # Programas / misiones
    "international space station": EntityType.PROGRAMA,
    "artemis program": EntityType.PROGRAMA,
    "artemis accords": EntityType.EVENTO,
    "outer space treaty": EntityType.EVENTO,
    "satellite constellation": EntityType.PROGRAMA,
    "tiangong": EntityType.PROGRAMA,
    "global positioning system": EntityType.PROGRAMA,
    # Tecnologías / conceptos IA
    "artificial intelligence": EntityType.TECNOLOGIA,
    "ai": EntityType.TECNOLOGIA,
    "natural language processing": EntityType.TECNOLOGIA,
    "computer vision": EntityType.TECNOLOGIA,
    "internet of things": EntityType.TECNOLOGIA,
    "large language models": EntityType.TECNOLOGIA,
    "foundation models": EntityType.TECNOLOGIA,
    "language models": EntityType.TECNOLOGIA,
    "chatbots": EntityType.TECNOLOGIA,
    "virtual assistants": EntityType.TECNOLOGIA,
    "digital twins": EntityType.TECNOLOGIA,
    "quantum computing": EntityType.TECNOLOGIA,
    "cloud computing": EntityType.TECNOLOGIA,
    "satellite": EntityType.TECNOLOGIA,
    "satellites": EntityType.TECNOLOGIA,
    "nanosatellite": EntityType.TECNOLOGIA,
    "remote sensing": EntityType.TECNOLOGIA,
    "synthetic aperture radar": EntityType.TECNOLOGIA,
    "unmanned aerial vehicles": EntityType.TECNOLOGIA,
    "reusable rockets": EntityType.TECNOLOGIA,
    "electric propulsion": EntityType.TECNOLOGIA,
    "space situational awareness": EntityType.TECNOLOGIA,
    "earth observation": EntityType.TECNOLOGIA,
    "satellite surveillance": EntityType.TECNOLOGIA,
    "satellite reconnaissance": EntityType.TECNOLOGIA,
    # Seguridad espacial / territorial
    "space debris": EntityType.CONFLICTO,
    "orbital debris": EntityType.CONFLICTO,
    "space junk": EntityType.CONFLICTO,
    "cybersecurity": EntityType.SECTOR,
    "cyber attacks": EntityType.CONFLICTO,
    "cyberwarfare": EntityType.CONFLICTO,
    "electronic warfare": EntityType.CONFLICTO,
    "hybrid warfare": EntityType.CONFLICTO,
    "information warfare": EntityType.CONFLICTO,
    "signal interference": EntityType.CONFLICTO,
    "jamming": EntityType.CONFLICTO,
    "spoofing": EntityType.CONFLICTO,
    "space security": EntityType.SECTOR,
    "national security": EntityType.SECTOR,
    "digital sovereignty": EntityType.SECTOR,
    "space sovereignty": EntityType.SECTOR,
    "sovereignty": EntityType.SECTOR,
    "defense": EntityType.SECTOR,
    "military intelligence": EntityType.SECTOR,
    # Sectores (EN)
    "agriculture": EntityType.SECTOR,
    "mining": EntityType.SECTOR,
    "energy": EntityType.SECTOR,
    "telecommunications": EntityType.SECTOR,
    "transportation": EntityType.SECTOR,
    "health": EntityType.SECTOR,
    "education": EntityType.SECTOR,
    "livestock": EntityType.SECTOR,
    "fishing": EntityType.SECTOR,
    "risk management": EntityType.SECTOR,
    "civil protection": EntityType.SECTOR,
    "civil defense": EntityType.SECTOR,
}

# Ids canónicos reutilizados por el gazetteer y por el mapa de alias (evita
# duplicar literales entre ES y EN: "space debris" → _CANONICO_BASURA_ESPACIAL).
_CANONICO_BASURA_ESPACIAL = "basura espacial"
_CANONICO_IA = "inteligencia artificial"
_CANONICO_ESTADOS_UNIDOS = "estados unidos"
_CANONICO_MODELOS_LENGUAJE = "modelos de lenguaje"


#: Alias de entidad (linking es↔en): forma normalizada → id canónico.
#: El corpus es mayormente inglés pero las consultas del reto son en español;
#: este mapa hace que "space debris" y "basura espacial" resuelvan al MISMO
#: nodo del grafo (mismo id canónico), sin depender del idioma de la consulta.
ALIASES: Dict[str, str] = {
    # Basura espacial
    "space debris": _CANONICO_BASURA_ESPACIAL,
    "orbital debris": _CANONICO_BASURA_ESPACIAL,
    "space junk": _CANONICO_BASURA_ESPACIAL,
    # Estación espacial
    "international space station": "estacion espacial internacional",
    # Órbitas
    "low earth orbit": "orbita terrestre baja",
    "leo": "orbita terrestre baja",
    "geostationary orbit": "orbita geoestacionaria",
    # IA
    "artificial intelligence": _CANONICO_IA,
    "ai": _CANONICO_IA,
    "ia": _CANONICO_IA,
    "machine learning": "aprendizaje automatico",
    "deep learning": "aprendizaje profundo",
    "natural language processing": "procesamiento de lenguaje natural",
    "computer vision": "vision por computador",
    "internet of things": "internet de las cosas",
    "large language models": _CANONICO_MODELOS_LENGUAJE,
    "language models": _CANONICO_MODELOS_LENGUAJE,
    "foundation models": "modelos fundacionales",
    "digital twins": "gemelos digitales",
    "quantum computing": "computacion cuantica",
    "cloud computing": "computacion en la nube",
    "chatbots": "chatbot",
    "virtual assistants": "asistentes virtuales",
    "satellite": "satelite",
    "satellites": "satelite",
    "nanosatellite": "nanosatelite",
    # Países / regiones
    "united states": _CANONICO_ESTADOS_UNIDOS,
    "usa": _CANONICO_ESTADOS_UNIDOS,
    "united nations": "organizacion de las naciones unidas",
    "latin america": "america latina",
    "central america": "centroamerica",
    "the caribbean": "el caribe",
    "antarctica": "antartida",
    "south korea": "corea del sur",
    "united arab emirates": "emiratos arabes unidos",
    "european union": "union europea",
    "nato": "otan",
    # Organizaciones
    "mexican space agency": "agencia espacial mexicana",
    "brazilian space agency": "agencia espacial brasilena",
    "european space agency": "agencia espacial europea",
    "national aeronautics and space administration": "nasa",
    "space agencies": "agencias espaciales",
    # Programas / eventos
    "artemis program": "programa artemisa",
    "artemis accords": "acuerdos artemisa",
    "outer space treaty": "tratado del espacio ultraterrestre",
    "satellite constellation": "constelacion de satelites",
    # Seguridad
    "space security": "seguridad espacial",
    "national security": "seguridad nacional",
    "cybersecurity": "ciberseguridad",
    "cyber attacks": "ataques ciberneticos",
    "cyberwarfare": "ciberguerra",
    "electronic warfare": "guerra electronica",
    "hybrid warfare": "guerra hibrida",
    "information warfare": "guerra de informacion",
    "signal interference": "interferencia de senales",
    "digital sovereignty": "soberania digital",
    "space sovereignty": "soberania espacial",
    "military intelligence": "inteligencia militar",
    "satellite surveillance": "vigilancia satelital",
    "satellite reconnaissance": "reconocimiento satelital",
    # Tecnologías
    "remote sensing": "teleobservacion",
    "synthetic aperture radar": "radar de apertura sintetica",
    "unmanned aerial vehicles": "vehiculos aereos no tripulados",
    "global positioning system": "gps",
}


@dataclass
class RegexEntityExtractorConfig:
    """Configuración de la estrategia NER simbólica.

    ``gazetteer_adicional`` permite extender el vocabulario sin tocar la
    clase (OCP): mapeo forma léxica → tipo.
    ``detectar_nombres_propios`` habilita el patrón genérico de
    mayúsculas para entidades no catalogadas (p. ej. "Agencia Espacial
    Mexicana" si no estuviera en el gazetteer).
    """

    gazetteer_adicional: Dict[str, EntityType] | None = None
    detectar_nombres_propios: bool = True

    def gazetteer_completo(self) -> Dict[str, EntityType]:
        """Fusiona el gazetteer base con el adicional (adicional gana)."""
        if not self.gazetteer_adicional:
            return dict(GAZETTEER)
        fusion = dict(GAZETTEER)
        fusion.update(self.gazetteer_adicional)
        return fusion


# Patrón de nombres propios: 2+ palabras capitalizadas consecutivas
# (p. ej. "Agencia Espacial Mexicana", "Fuerza Aérea Brasileña").
_PATRON_NOMBRE_PROPIO = re.compile(
    r"\b(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}\s+)+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}\b"
)

_ESPACIO = re.compile(r"\s+")


@EntityExtractorFactory.register("regex-gazetteer")
class RegexEntityExtractor(EntityExtractor):
    """NER determinista por gazetteer + nombres propios (sin LLM)."""

    name = "regex-gazetteer"

    def __init__(self, config: RegexEntityExtractorConfig | None = None) -> None:
        self.config = config or RegexEntityExtractorConfig()
        self._gazetteer = self.config.gazetteer_completo()
        # Precompila las entradas NORMALIZADAS (sin acentos), de más largas a
        # más cortas: el matching por spans deja ganar a "inteligencia
        # artificial" sobre "inteligencia". Normalizar la clave garantiza que
        # "órbita terrestre baja" (con tilde) coincida con el texto
        # normalizado "orbita terrestre baja".
        patrones_vistos: set[str] = set()
        self._patrones: List[Tuple[Pattern[str], str, EntityType]] = []
        for nombre, tipo in sorted(
            self._gazetteer.items(),
            key=lambda item: len(item[0].split()),
            reverse=True,
        ):
            normalizada = _ESPACIO.sub(" ", normalizar_id_entidad(nombre))
            if not normalizada or normalizada in patrones_vistos:
                continue
            patrones_vistos.add(normalizada)
            self._patrones.append(
                (re.compile(rf"(?<!\w){re.escape(normalizada)}(?!\w)"), nombre, tipo)
            )

    # -- NER ---------------------------------------------------------------

    def extraer_entidades(self, texto: str) -> List[Entity]:
        """Entidades detectadas en ``texto``, deduplicadas por id canónico.

        Algoritmo: normaliza el texto (minúsculas sin acentos), busca cada
        patrón del gazetteer y deja SOLO los matches no solapados (ganan los
        de más palabras); luego añade los nombres propios genéricos que no
        colisionen con spans ya ocupados.
        """
        if not texto:
            return []
        ntext = _ESPACIO.sub(" ", normalizar_id_entidad(texto))
        spans: List[Tuple[int, int, str, EntityType, str]] = []
        for patron, nombre, tipo in self._patrones:
            for m in patron.finditer(ntext):
                if not self._solapado(spans, m.start(), m.end()):
                    spans.append((m.start(), m.end(), nombre, tipo, nombre))

        if self.config.detectar_nombres_propios:
            spans = self._agregar_nombres_propios(texto, spans)

        spans.sort(key=lambda s: s[0])

        # Deduplicación por id canónico: la misma entidad puede aparecer en
        # varios spans ("Perú y PERÚ..."); se conserva la primera y se
        # acumulan las variantes léxicas.
        por_id: Dict[str, Entity] = {}
        for s in spans:
            entidad = self._a_entidad(s)
            if entidad.id in por_id:
                if entidad.nombre not in por_id[entidad.id].variantes:
                    por_id[entidad.id].variantes.append(entidad.nombre)
            else:
                por_id[entidad.id] = entidad
        return list(por_id.values())

    def _agregar_nombres_propios(
        self, texto: str, spans: List[Tuple[int, int, str, EntityType, str]]
    ) -> List[Tuple[int, int, str, EntityType, str]]:
        """Añade nombres propios genéricos (2+ capitalizadas) no cubiertos.

        Las posiciones se calculan sobre el texto NORMALIZADO (las mismas
        que usa el gazetteer) mapeando el fragmento original a su versión
        normalizada, de modo que la comprobación de solapamiento sea exacta.
        """
        for m in _PATRON_NOMBRE_PROPIO.finditer(texto):
            forma = m.group(0).strip()
            if len(forma.split()) < 2:
                continue
            # Recorta palabras de función de los bordes ("La Fuerza Aérea..."
            # → "Fuerza Aérea") y exige que queden 2+ palabras significativas.
            partes = forma.split()
            n_prefijo = _ESPACIO.sub(" ", normalizar_id_entidad(texto[: m.start()])).strip()
            while partes and partes[0].lower() in _PALABRAS_FUNCION:
                n_prefijo = (
                    n_prefijo + " " + _ESPACIO.sub(" ", normalizar_id_entidad(partes.pop(0)))
                ).strip()
            while partes and partes[-1].lower() in _PALABRAS_FUNCION:
                partes.pop()
            if len(partes) < 2:
                continue
            forma = " ".join(partes)
            n_inicio = len(n_prefijo)
            n_fin = n_inicio + len(_ESPACIO.sub(" ", normalizar_id_entidad(forma)).strip())
            if not self._solapado(spans, n_inicio, n_fin):
                spans.append((n_inicio, n_fin, forma, EntityType.OTRO, forma))
        return spans

    @staticmethod
    def _solapado(
        spans: List[Tuple[int, int, str, EntityType, str]], inicio: int, fin: int
    ) -> bool:
        """True si el rango [inicio, fin) solapa algún span ya registrado."""
        return any(not (fin <= s_inicio or inicio >= s_fin) for s_inicio, s_fin, *_ in spans)

    @staticmethod
    def _a_entidad(
        span: Tuple[int, int, str, EntityType, str],
    ) -> Entity:
        """Convierte un span en :class:`Entity` con id canónico y variantes.

        El id canónico se resuelve con el mapa de alias es↔en (p. ej.
        "space debris" → "basura espacial") para que consultas y corpus en
        idiomas distintos apunten al mismo nodo del grafo.
        """
        _, _, forma, tipo, _ = span
        id_canonico = ALIASES.get(normalizar_id_entidad(forma), normalizar_id_entidad(forma))
        return Entity(
            id=id_canonico,
            nombre=forma,
            tipo=tipo,
            variantes=[forma],
        )


# Palabras de función que desactivan el falso positivo "X Y" al inicio de
# oración (artículos, preposiciones, conjunciones, verbos copulativos...).
_PALABRAS_FUNCION = frozenset(
    {
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        "de", "del", "en", "con", "por", "para", "sobre", "entre",
        "y", "o", "e", "u", "que", "al", "a", "se", "su", "sus",
        "como", "tras", "durante", "desde", "hasta", "según", "the", "of",
        "an", "and", "in", "on", "with", "for", "to", "from", "is", "are",
    }
)
