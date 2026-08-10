"""Serializador GraphML (Pure Fabrication, GRASP).

Clase técnica que NO representa un concepto del dominio del reto pero
resuelve un problema real: convertir :class:`KnowledgeGraph` a GraphML
válido usando SOLO la stdlib (``xml.etree.ElementTree``), de modo que el
archivo sea cargable con NetworkX u otra herramienta SIN dependencias
adicionales (requisito duro de la Sección 7).

Garantías del formato emitido:
- Nodos con atributo ``tipo`` (EntityType) y ``nombre`` (forma léxica).
- Una arista por tripleta con ``relacion``, ``doc_id``, ``chunk_id``,
  ``confianza`` y ``evidencia`` (trazabilidad de la Sección 7.3).
- ``id`` único por arista (las tripletas paralelas no colisionan).
- Declaración ``<key>`` para cada atributo (GraphML exige declarar los
  atributos antes de usarlos).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

from src.knowledge_graph.graph.knowledge_graph import KnowledgeGraph

_NS = "http://graphml.graphdrawing.org/xmlns"
_NSMAP = {"": _NS}

#: Atributos de nodo y arista que se declaran en el encabezado GraphML.
#: ``sujeto``/``objeto`` conservan la orientación canónica de la tripleta
#: (NetworkX normaliza el orden de las aristas no dirigidas al leerlas).
_ATTR_NODO = (("tipo", "string"), ("nombre", "string"))
_ATTR_ARISTA = (
    ("relacion", "string"),
    ("sujeto", "string"),
    ("objeto", "string"),
    ("doc_id", "string"),
    ("chunk_id", "string"),
    ("confianza", "double"),
    ("evidencia", "string"),
)


class GraphMLSerializer:
    """Convierte :class:`KnowledgeGraph` ⇄ GraphML (texto o archivo)."""

    # -- Escritura ----------------------------------------------------------

    def serializar(self, grafo: KnowledgeGraph) -> str:
        """Devuelve el GraphML de ``grafo`` como string (XML declarado UTF-8)."""
        root = ET.Element("graphml", xmlns=_NS)

        # Declaración de atributos (obligatoria en GraphML; el estándar usa
        # nombres de atributo XML con punto: attr.name / attr.type / for).
        for i, (nombre, tipo) in enumerate(_ATTR_NODO):
            ET.SubElement(
                root,
                "key",
                id=f"kn{i}",
                **{"for": "node", "attr.name": nombre, "attr.type": tipo},
            )
        for i, (nombre, tipo) in enumerate(_ATTR_ARISTA):
            ET.SubElement(
                root,
                "key",
                id=f"ka{i}",
                **{"for": "edge", "attr.name": nombre, "attr.type": tipo},
            )

        graph = ET.SubElement(root, "graph", id="G", edgedefault="undirected")

        for entidad in grafo.entidades():
            nodo = ET.SubElement(graph, "node", id=entidad.id)
            ET.SubElement(nodo, "data", key="kn0").text = entidad.tipo.value
            ET.SubElement(nodo, "data", key="kn1").text = entidad.nombre

        for i, tripleta in enumerate(grafo.tripletas()):
            arista = ET.SubElement(
                graph,
                "edge",
                id=f"e{i}",
                source=tripleta.sujeto,
                target=tripleta.objeto,
            )
            ET.SubElement(arista, "data", key="ka0").text = tripleta.relacion.value
            ET.SubElement(arista, "data", key="ka1").text = tripleta.sujeto
            ET.SubElement(arista, "data", key="ka2").text = tripleta.objeto
            ET.SubElement(arista, "data", key="ka3").text = tripleta.doc_id
            ET.SubElement(arista, "data", key="ka4").text = tripleta.chunk_id
            ET.SubElement(arista, "data", key="ka5").text = f"{tripleta.confianza:.6f}"
            if tripleta.evidencia:
                ET.SubElement(arista, "data", key="ka6").text = tripleta.evidencia

        # ET.indent (3.9+) para un archivo legible; compatible hacia atrás.
        try:
            ET.indent(root, space="  ")
        except AttributeError:  # pragma: no cover - Python < 3.9
            pass

        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def escribir(self, grafo: KnowledgeGraph, ruta: Union[str, Path]) -> Path:
        """Escribe el GraphML de ``grafo`` en ``ruta`` (crea el directorio).

        Returns:
            La ruta escrita (``Path``), para encadenar.
        """
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(self.serializar(grafo), encoding="utf-8")
        return ruta
