"""
Paquete raíz del pipeline de ingesta RAG — CODEFEST AD ASTRA 2026.

Agrupa los submódulos funcionales del pipeline:
- ``extractors``: extracción de texto por formato (patrón Factory Method).
- ``chunking``: estrategias de fragmentación (patrón Strategy).
- ``cleaning``: limpieza y normalización de texto.
- ``metadata``: construcción de metadata obligatoria por fragmento.
- ``validation``: validaciones previas a la persistencia.
- ``persistence``: repositorio de fragmentos (MongoDB).
- ``pipeline``: orquestación del flujo completo.
"""
