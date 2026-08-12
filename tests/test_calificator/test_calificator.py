"""Tests del calificador (Sección 10): NDCG@10, F1@3, cargas e informe.

Deterministas y sin red: métricas con valores calculados a mano y
resultados/ground truth sintéticos (esquema 9.3.1).
"""

from __future__ import annotations

import json

import pytest

from src.calificator.calificator import Calificator
from src.calificator.loader import cargar_ground_truth, cargar_resultados
from src.calificator.metrics import (
    dcg,
    f1_3,
    fuente_de_doc_id,
    idcg,
    ndcg_10,
    normalizar_texto,
    relevancia_de,
)
from src.calificator.models import (
    DocumentoResultado,
    FragmentoRelevante,
    FragmentoResultado,
    GroundTruth,
    GroundTruthConsulta,
    ResultadoConsulta,
)


# -- NDCG@10 ------------------------------------------------------------------

def test_dcg_valor_conocido():
    # r = [3, 2, 3] → 3/log2(2) + 2/log2(3) + 3/log2(4)
    esperado = 3 / 1.0 + 2 / 1.584962500721156 + 3 / 2.0
    assert dcg([3, 2, 3], 10) == pytest.approx(esperado)


def test_dcg_ignora_relevancia_cero():
    assert dcg([0, 0, 0], 10) == 0.0
    assert dcg([3, 0, 3], 10) == pytest.approx(3 / 1.0 + 3 / 2.0)


def test_idcg_ordena_descendente():
    # Relevancias posibles [1, 3, 2] → ideal [3, 2, 1]
    assert idcg([1, 3, 2], 10) == dcg([3, 2, 1], 10)


def test_ndcg_10_perfecto():
    # Ranking entregado ya es el ideal → NDCG = 1.0
    assert ndcg_10([3, 2, 1, 0], [1, 2, 3]) == pytest.approx(1.0)


def test_ndcg_10_sin_relevantes_es_1():
    assert ndcg_10([0, 0, 0], []) == 1.0


def test_ndcg_10_mitad():
    # Ideal [3,2,1]; entregado [0,3,2] → NDCG = DCG([0,3,2])/DCG([3,2,1])
    entregado = dcg([0, 3, 2], 10)
    ideal = idcg([1, 2, 3], 10)
    assert ndcg_10([0, 3, 2, 0], [1, 2, 3]) == pytest.approx(entregado / ideal)


# -- Emparejamiento de fragmentos ---------------------------------------------

def test_normalizar_texto():
    assert normalizar_texto("  Órbita  Terrestre! ") == "orbita terrestre"
    assert normalizar_texto("Estados Unidos") == "estados unidos"


def test_relevancia_de_exacta():
    relevantes = [FragmentoRelevante(texto="La ESA coopera con la NASA.", relevancia=3)]
    assert relevancia_de("La ESA coopera con la NASA.", relevantes) == 3


def test_relevancia_de_contencion():
    relevantes = [
        FragmentoRelevante(
            texto="La ESA coopera con la NASA en la órbita terrestre baja.", relevancia=2
        )
    ]
    assert relevancia_de("La ESA coopera con la NASA", relevantes) == 2


def test_relevancia_de_sin_match():
    relevantes = [FragmentoRelevante(texto="Texto irrelevante.", relevancia=3)]
    assert relevancia_de("Otro texto completamente distinto.", relevantes) == 0


# -- F1@3 ---------------------------------------------------------------------

def test_fuente_de_doc_id():
    assert fuente_de_doc_id(r"F1_IA\AI_Index\pdfs\reporte.pdf") == "reporte.pdf"
    assert fuente_de_doc_id("F2_Seguridad/CSIS/articulos/doc.json") == "doc.json"
    assert fuente_de_doc_id("DOC-042") == "DOC-042"


def test_f1_3_perfecto():
    precision, recall, f1 = f1_3(
        [r"F1\AI_Index\reporte.pdf", "DOC-017", r"F3\CEEEP\articulo.json"],
        ["reporte.pdf"],
    )
    assert precision == pytest.approx(1 / 3)
    assert recall == pytest.approx(1.0)  # min(|D*|, 3) = 1
    assert f1 == pytest.approx(2 * (1 / 3) * 1.0 / (1 / 3 + 1.0))


def test_f1_3_sin_aciertos_es_cero():
    precision, recall, f1 = f1_3(["a.pdf", "b.pdf", "c.pdf"], ["x.pdf"])
    assert (precision, recall, f1) == (0.0, 0.0, 0.0)


def test_f1_3_recall_limita_a_min_3():
    # 3 relevantes, 2 acertados → recall = 2/3
    precision, recall, f1 = f1_3(["a.pdf", "b.pdf", "c.pdf"], ["a.pdf", "b.pdf", "d.pdf"])
    assert recall == pytest.approx(2 / 3)


# -- Loader -------------------------------------------------------------------

def test_cargar_resultados_jsonl(tmp_path):
    ruta = tmp_path / "resultados.jsonl"
    ruta.write_text(
        json.dumps(
            {
                "query_id": "q001",
                "documents": [{"rank": 1, "doc_id": "DOC-042"}],
                "fragments": [
                    {"rank": 1, "chunk_id": "DOC-042-chunk-007", "doc_id": "DOC-042",
                     "text": "Texto del fragmento."}
                ],
            }
        ),
        encoding="utf-8",
    )
    resultados = cargar_resultados(ruta)
    assert len(resultados) == 1
    assert resultados[0].query_id == "q001"
    assert resultados[0].documents[0].doc_id == "DOC-042"
    assert resultados[0].fragments[0].text == "Texto del fragmento."


def test_cargar_ground_truth_json(tmp_path):
    ruta = tmp_path / "gt.json"
    ruta.write_text(
        json.dumps(
            {
                "q001": {
                    "fragmentos": [
                        {"texto": "Texto A", "relevancia": 3},
                        "Texto B",
                    ],
                    "documentos": ["archivo.pdf"],
                }
            }
        ),
        encoding="utf-8",
    )
    gt = cargar_ground_truth(ruta)
    assert gt.ids == ["q001"]
    consulta = gt.para("q001")
    assert consulta.fragmentos[0].relevancia == 3
    assert consulta.fragmentos[1].relevancia == 1.0  # string → 1.0
    assert consulta.documentos == ["archivo.pdf"]


def test_cargar_ground_truth_jsonl(tmp_path):
    ruta = tmp_path / "gt.jsonl"
    ruta.write_text(
        json.dumps({"query_id": "q002", "fragmentos": ["X"], "documentos": ["y.pdf"]})
        + "\n",
        encoding="utf-8",
    )
    gt = cargar_ground_truth(ruta)
    assert gt.ids == ["q002"]


# -- Calificator (integración) -------------------------------------------------

def _resultado(query_id: str, doc_ids, fragmentos_texto) -> ResultadoConsulta:
    return ResultadoConsulta(
        query_id=query_id,
        documents=[
            DocumentoResultado(rank=i + 1, doc_id=doc_id)
            for i, doc_id in enumerate(doc_ids)
        ],
        fragments=[
            FragmentoResultado(rank=i + 1, chunk_id=f"{query_id}-chunk-{i:03d}",
                               doc_id=doc_ids[i % len(doc_ids)], text=texto)
            for i, texto in enumerate(fragmentos_texto)
        ],
    )


def test_calificator_informe_completo():
    resultados = [
        _resultado(
            "q001",
            [r"F1\AI_Index\reporte.pdf", "DOC-017", "DOC-091"],
            ["El AI Index es un informe anual de Stanford."]
            + [f"Texto distinto {i}." for i in range(9)],
        )
    ]
    gt = GroundTruth(
        consultas={
            "q001": GroundTruthConsulta(
                query_id="q001",
                fragmentos=[
                    FragmentoRelevante(
                        texto="El AI Index es un informe anual de Stanford.", relevancia=3
                    )
                ],
                documentos=["reporte.pdf"],
            )
        }
    )
    informe = Calificator().calificar(resultados, gt)
    assert informe.consultas_evaluadas == 1
    consulta = informe.por_consulta[0]
    # NDCG: fragmento 1 relevante (3), resto 0 → DCG = 3/log2(2) = 3; IDCG = 3 → 1.0
    assert consulta.ndcg10 == pytest.approx(1.0)
    # F1@3: 1 de 3 docs acertados, recall = 1/1 → P=1/3, R=1, F1 = 2*(1/3)/(4/3) = 0.5
    assert consulta.f1_3 == pytest.approx(0.5)
    assert informe.ndcg10_media == pytest.approx(1.0)
    assert informe.f1_3_media == pytest.approx(0.5)
    # Desglose legible: el top incluye texto y relevancia.
    assert consulta.fragmentos_top[0]["relevancia"] == 3.0
    assert consulta.fragmentos_top[0]["texto"].startswith("El AI Index")


def test_calificator_omite_consultas_sin_ground_truth():
    resultados = [_resultado("q999", ["a.pdf", "b.pdf", "c.pdf"], ["Texto."] * 10)]
    gt = GroundTruth(consultas={})
    informe = Calificator().calificar(resultados, gt)
    assert informe.consultas_evaluadas == 0
    assert informe.ndcg10_media == 0.0
    assert informe.f1_3_media == 0.0


def test_informe_a_dict_y_markdown():
    resultados = [
        _resultado("q001", ["a.pdf", "b.pdf", "c.pdf"], ["Texto A."] * 10)
    ]
    gt = GroundTruth(
        consultas={
            "q001": GroundTruthConsulta(
                query_id="q001", fragmentos=[], documentos=[]
            )
        }
    )
    informe = Calificator().calificar(resultados, gt)
    dato = informe.a_dict()
    assert "ndcg10_media" in dato and "por_consulta" in dato
    md = informe.a_markdown()
    assert "NDCG@10" in md and "q001" in md