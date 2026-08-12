"""Adapters que implementan :class:`Retriever` (patrón Adapter, Sección 8.5).

Traducen cada canal de evidencia al contrato común (:class:`ScoredChunk`)
para que el :class:`RetrievalOrchestrator` y el motor de fusión traten al
grafo de conocimiento y al índice FAISS de forma uniforme (DIP/OCP):

- :class:`VectorIndexAdapter` → canal vectorial (encoder + FAISS).
- :class:`GraphIndexAdapter` → canal simbólico (NER + grafo + vecindario).

Ninguno invoca LLMs; ambos devuelven listas ya ordenadas por score.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

from src.encoders.base import EncoderStrategy
from src.knowledge_graph.extract.base import EntityExtractor
from src.knowledge_graph.graph.repository import GraphQuery
from src.knowledge_graph.models import Query, ScoredChunk
from src.knowledge_graph.retrieval.base import Retriever

logger = logging.getLogger(__name__)


class VectorIndexAdapter(Retriever):
    """Canal vectorial: codifica la consulta y busca en FAISS (Sección 8.1-8.2).

    Reutiliza las etapas puras existentes del pipeline de recuperación
    (``encode_query`` + ``search_faiss``): el adapter solo traduce
    :class:`SearchHit` → :class:`ScoredChunk` (bajo acoplamiento: no
    conoce el interior del índice ni del encoder, solo sus interfaces).
    """

    def __init__(
        self,
        encoder: EncoderStrategy,
        index,
        metadata: Sequence[dict],
        encoder_name: str,
    ) -> None:
        self._encoder = encoder
        self._index = index
        self._metadata = metadata
        self._encoder_name = encoder_name

    @property
    def origen(self) -> str:
        return self._encoder_name

    def retrieve(self, q: Query, k: int) -> List[ScoredChunk]:
        """Embebe ``q`` y devuelve los ``k`` chunks más similares por coseno."""
        from src.retrieval.faiss_search import search_faiss
        from src.retrieval.query_encoder import encode_query

        vector = encode_query(self._encoder, q.texto)
        hits = search_faiss(
            self._index,
            vector,
            self._metadata,
            encoder_name=self._encoder_name,
            k=k,
        )
        return [
            ScoredChunk(
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                score=h.score,
                origen=self._encoder_name,
                metadata=dict(h.metadata),
            )
            for h in hits
        ]


class GraphIndexAdapter(Retriever):
    """Canal simbólico: NER sobre la consulta → entidades → vecindario → chunks.

    Flujo (Sección 8.5): se aplica el MISMO :class:`EntityExtractor` usado
    al construir el grafo, se buscan las entidades en el grafo vía
    :class:`GraphQuery`, se expande a vecinos de primer orden y se colectan
    los ``chunk_id`` de las tripletas involucradas.

    Scoring documentado (heurística sin LLM): cada tripleta DIRECTA de una
    entidad de la consulta aporta ``peso_directo`` a su chunk; cada tripleta
    de un VECINO de primer orden aporta ``peso_vecino``. El score de un
    chunk es la suma de evidencias que lo sustentan.
    """

    def __init__(
        self,
        extractor: EntityExtractor,
        grafo: GraphQuery,
        peso_directo: float = 1.0,
        peso_vecino: float = 0.5,
    ) -> None:
        self._extractor = extractor
        self._grafo = grafo
        self._peso_directo = peso_directo
        self._peso_vecino = peso_vecino

    @property
    def origen(self) -> str:
        return "grafo"

    def _evidencias(self, q: Query) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Evidencia por chunk: tripletas directas (peso_directo) + de vecinos.

        Returns:
            ``(evidencias, doc_ids)``: score acumulado por ``chunk_id`` y el
            ``doc_id`` de cada chunk (Information Expert: el adapter sabe
            resolver entidades → vecinos → chunks).
        """
        evidencias: Dict[str, float] = {}
        doc_ids: Dict[str, str] = {}
        for entidad in self._extractor.extraer_entidades(q.texto):
            if not self._grafo.tiene(entidad.id):
                continue
            for tripleta in self._grafo.tripletas_de(entidad.id):
                evidencias[tripleta.chunk_id] = (
                    evidencias.get(tripleta.chunk_id, 0.0) + self._peso_directo
                )
                doc_ids[tripleta.chunk_id] = tripleta.doc_id
            for vecino in self._grafo.vecinos_primer_orden(entidad.id):
                for tripleta in self._grafo.tripletas_de(vecino):
                    evidencias[tripleta.chunk_id] = (
                        evidencias.get(tripleta.chunk_id, 0.0) + self._peso_vecino
                    )
                    doc_ids[tripleta.chunk_id] = tripleta.doc_id
        return evidencias, doc_ids

    def retrieve(self, q: Query, k: int) -> List[ScoredChunk]:
        """Chunks vinculados a las entidades de ``q`` y a sus vecinos.

        Returns:
            Hasta ``k`` :class:`ScoredChunk` ordenados por evidencia
            (score) descendente; vacío si la consulta no resuelve ninguna
            entidad del grafo (el grafo "no opina", no inventa).
        """
        evidencias, doc_ids = self._evidencias(q)
        resultados = [
            ScoredChunk(
                chunk_id=cid,
                doc_id=doc_ids[cid],
                score=score,
                origen=self.origen,
                metadata={"fuente": "grafo"},
            )
            for cid, score in evidencias.items()
        ]
        resultados.sort(key=lambda c: (-c.score, c.chunk_id))
        return resultados[:k]

    def explicar_camino(
        self,
        q: Query,
        max_tripletas: int = 100,
        max_chunks: int = 50,
        chunk_ids_objetivo: Optional[List[str]] = None,
    ) -> dict:
        """Camino por el grafo para ``q`` (auditoría de la Sección 8.5).

        Devuelve la traza que conecta la consulta con los chunks de
        evidencia: entidades detectadas por el NER → nodos del grafo que
        matchean → vecinos de primer orden → tripletas involucradas → chunks
        con su score de evidencia (justificación simbólica, sin LLM).

        Si se pasa ``chunk_ids_objetivo`` (p. ej. el top-k fusionado), añade
        ``camino_por_top_k``: para cada chunk objetivo, las tripletas del
        grafo que lo sustentan (las que tienen ese chunk como evidencia y
        conectan entidades de la consulta o sus vecinos).

        Args:
            max_tripletas: tope de tripletas del camino global (las de mayor
                evidencia; evita JSONs enormes con entidades muy frecuentes).
            max_chunks: tope de chunks de evidencia globales incluidos.
            chunk_ids_objetivo: chunks del resultado fusionado a explicar.
        """
        entidades = self._extractor.extraer_entidades(q.texto)
        en_grafo = [e for e in entidades if self._grafo.tiene(e.id)]
        relevantes = self._entidades_relevantes(en_grafo)

        vecinos: Dict[str, List[str]] = {
            e.id: sorted(self._grafo.vecinos_primer_orden(e.id)) for e in en_grafo
        }

        evidencias, _ = self._evidencias(q)

        tripletas_vistas: Dict[Tuple[str, str, str, str], dict] = {}
        for e in en_grafo:
            self._acumular_tripletas(e.id, tripletas_vistas)
            for vecino in vecinos[e.id]:
                self._acumular_tripletas(vecino, tripletas_vistas)

        # Camino priorizado: las tripletas de los chunks con más evidencia
        # primero (es el "camino escogido" por el scoring del canal).
        tripletas_ordenadas = sorted(
            tripletas_vistas.values(),
            key=lambda t: (
                -evidencias.get(t["chunk_id"], 0.0),
                t["sujeto"],
                t["relacion"],
                t["objeto"],
                t["chunk_id"],
            ),
        )[:max_tripletas]

        camino_por_top_k: List[dict] = []
        if chunk_ids_objetivo:
            for cid in chunk_ids_objetivo:
                camino_por_top_k.append(
                    self._camino_para_chunk(cid, relevantes, evidencias)
                )

        return {
            "entidades_consulta": [
                {"id": e.id, "tipo": e.tipo.value, "nombre": e.nombre} for e in entidades
            ],
            "entidades_en_grafo": [e.id for e in en_grafo],
            "vecinos_primer_orden": vecinos,
            "tripletas": tripletas_ordenadas,
            "chunks_evidencia": dict(
                sorted(evidencias.items(), key=lambda kv: (-kv[1], kv[0]))[:max_chunks]
            ),
            "camino_por_top_k": camino_por_top_k,
            "aportó": bool(evidencias),
        }

    def _acumular_tripletas(
        self, entidad_id: str, destino: Dict[Tuple[str, str, str, str], dict]
    ) -> None:
        """Registra en ``destino`` las tripletas que involucran a ``entidad_id``."""
        for t in self._grafo.tripletas_de(entidad_id):
            destino.setdefault(
                (t.sujeto, t.relacion.value, t.objeto, t.chunk_id),
                {
                    "sujeto": t.sujeto,
                    "relacion": t.relacion.value,
                    "objeto": t.objeto,
                    "doc_id": t.doc_id,
                    "chunk_id": t.chunk_id,
                    "confianza": t.confianza,
                },
            )

    def _entidades_relevantes(self, en_grafo: Sequence) -> set:
        """Ids de entidades relevantes: las de la consulta + sus vecinos.

        Information Expert: solo el adapter sabe qué entidades conectan con
        la consulta a través del grafo (el conjunto usado para explicar un
        chunk objetivo).
        """
        relevantes = {e.id for e in en_grafo}
        for e in en_grafo:
            relevantes.update(self._grafo.vecinos_primer_orden(e.id))
        return relevantes

    def _camino_para_chunk(
        self,
        chunk_id: str,
        relevantes: set,
        evidencias: Dict[str, float],
    ) -> dict:
        """Tripletas del grafo que sustentan ``chunk_id`` (si aporta evidencia)."""
        tripletas: List[dict] = []
        for t in self._grafo.tripletas_de_chunk(chunk_id):
            if t.sujeto in relevantes or t.objeto in relevantes:
                tripletas.append(
                    {
                        "sujeto": t.sujeto,
                        "relacion": t.relacion.value,
                        "objeto": t.objeto,
                        "doc_id": t.doc_id,
                        "confianza": t.confianza,
                    }
                )
        return {
            "chunk_id": chunk_id,
            "score_grafo": evidencias.get(chunk_id, 0.0),
            "tripletas": sorted(
                tripletas[:10], key=lambda t: (t["sujeto"], t["relacion"], t["objeto"])
            ),
        }
