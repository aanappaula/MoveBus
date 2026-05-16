"""RAG Retriever — recuperação de documentos relevantes para uma query.

Busca semântica na base vetorial e monta contexto consolidado para o LLM.
"""

import logging

from movebus.indexer.vector_store import VectorStore
from movebus.models import RetrievalResult, RetrievedDocument

logger = logging.getLogger(__name__)

_TOP_K = 5
_MIN_SCORE = 0.4


class BusInfoRetriever:
    """Recupera documentos relevantes da base vetorial para uma query.

    Monta contexto consolidado com metadados (nome da linha, fonte) para
    ser enviado ao LLM.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        """Inicializa o retriever com um VectorStore já configurado.

        Args:
            vector_store: Instância do VectorStore a ser consultada.
        """
        self._store = vector_store

    def retrieve(self, query: str) -> RetrievalResult:
        """Busca documentos relevantes para a query.

        Realiza busca semântica com top_k=5 e min_score=0.6. Monta contexto
        consolidado concatenando os documentos com separador '---' e
        cabeçalho de metadados por trecho.

        Args:
            query: Texto da pergunta do usuário.

        Returns:
            RetrievalResult com documentos recuperados, flag has_results e
            contexto consolidado. has_results=False quando nenhum documento
            atingir o score mínimo.
        """
        retrieved: list[RetrievedDocument] = self._store.search(
            query=query,
            top_k=_TOP_K,
            min_score=_MIN_SCORE,
        )

        if not retrieved:
            logger.info("Nenhum documento relevante encontrado para a query.")
            return RetrievalResult(
                documents=[], has_results=False, context=""
            )

        context = self._build_context(retrieved)
        logger.info(
            "%d documento(s) recuperado(s) para a query.", len(retrieved)
        )
        return RetrievalResult(
            documents=retrieved, has_results=True, context=context
        )

    @staticmethod
    def _build_context(documents: list[RetrievedDocument]) -> str:
        """Monta o contexto consolidado a partir dos documentos recuperados.

        Cada trecho é precedido por um cabeçalho com nome da linha e URL
        de origem, separados por '---'.

        Args:
            documents: Lista de documentos recuperados com score.

        Returns:
            String com o contexto consolidado.
        """
        parts: list[str] = []
        for rd in documents:
            doc = rd.document
            header = f"[Linha {doc.line_number} — {doc.line_name}] (fonte: {doc.source_url})"
            parts.append(f"{header}\n{doc.content}")

        return "\n---\n".join(parts)
