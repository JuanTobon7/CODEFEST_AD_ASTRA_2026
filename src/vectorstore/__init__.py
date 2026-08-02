"""
Persistencia de vectores en MongoDB + construcción de índices FAISS por
encoder (Sección 5, complemento de ``src.encoders``/``src.embeddings``).

Este paquete no importa por defecto los módulos que dependen de ``faiss``
(construcción de índices) para mantener ``vector_repository``/``models``
usables sin esa dependencia instalada, igual que ``src.encoders`` hace con
``sentence-transformers``.
"""
