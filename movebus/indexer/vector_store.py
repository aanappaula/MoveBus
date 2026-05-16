"""VectorStore — gerenciamento da base vetorial ChromaDB.

Responsável por gerar embeddings com sentence-transformers e armazenar/recuperar
documentos na base vetorial ChromaDB com suporte a deduplicação por hash SHA-256.
"""

import logging
import time
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from movebus.models import Document, IndexingResult, RetrievedDocument

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_COLLECTION_NAME = "movebus_documents"


class VectorStoreError(Exception):
    """Erro interno do VectorStore."""


class VectorStore:
    """Gerencia a base vetorial ChromaDB com embeddings multilinguais.

    Suporta inserção com deduplicação, busca semântica com filtro de score
    mínimo e reindexação completa da base.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        in_memory: bool = False,
    ) -> None:
        """Inicializa o VectorStore.

        Args:
            persist_directory: Diretório para persistência do ChromaDB.
                               Ignorado quando ``in_memory=True``.
            in_memory: Se True, usa cliente em memória (útil para testes).
        """
        logger.info("Carregando modelo de embedding '%s'…", _EMBEDDING_MODEL)
        self._model = SentenceTransformer(_EMBEDDING_MODEL)

        if in_memory:
            self._client = chromadb.Client()
        else:
            self._client = chromadb.PersistentClient(path=persist_directory)

        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore inicializado. Documentos na base: %d", self._collection.count()
        )

    # ------------------------------------------------------------------
    # Inserção
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """Insere documentos na base, ignorando duplicatas por hash SHA-256.

        Para cada documento, verifica se o ``content_hash`` já existe na
        coleção antes de inserir. Documentos duplicados são silenciosamente
        ignorados (Requirement 1.5).

        Args:
            documents: Lista de documentos a inserir.
        """
        if not documents:
            return

        # Recupera todos os hashes já presentes na coleção de uma só vez
        existing_ids = self._get_existing_ids()

        new_docs = [doc for doc in documents if doc.content_hash not in existing_ids]

        if not new_docs:
            logger.debug("Nenhum documento novo para inserir (todos duplicados).")
            return

        ids, embeddings, metadatas, contents = self._prepare_batch(new_docs)

        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=contents,
            )
            logger.info("%d documento(s) inserido(s) na base vetorial.", len(new_docs))
        except Exception as exc:
            raise VectorStoreError(f"Falha ao inserir documentos: {exc}") from exc

    # ------------------------------------------------------------------
    # Reindexação
    # ------------------------------------------------------------------

    def clear_and_reindex(self, documents: list[Document]) -> IndexingResult:
        """Limpa a base e reindexia todos os documentos.

        Usado para atualização completa da base de conhecimento
        (Requirement 8.2).

        Args:
            documents: Lista completa de documentos a indexar.

        Returns:
            IndexingResult com estatísticas da operação.
        """
        start = time.monotonic()
        errors: list[str] = []

        # Conta documentos antes de limpar
        previous_count = self._collection.count()

        # Remove a coleção e recria do zero
        try:
            self._client.delete_collection(_COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(f"Falha ao limpar a coleção: {exc}") from exc

        # Reinsere todos os documentos
        indexed = 0
        if documents:
            ids, embeddings, metadatas, contents = self._prepare_batch(documents)
            try:
                self._collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=contents,
                )
                indexed = len(documents)
            except Exception as exc:
                error_msg = f"Falha ao reindexar documentos: {exc}"
                logger.error(error_msg)
                errors.append(error_msg)

        elapsed = time.monotonic() - start
        logger.info(
            "Reindexação concluída: %d inseridos, %d removidos, %.2fs.",
            indexed,
            previous_count,
            elapsed,
        )

        return IndexingResult(
            documents_indexed=indexed,
            documents_removed=previous_count,
            execution_time_seconds=elapsed,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Busca
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.6,
    ) -> list[RetrievedDocument]:
        """Busca semântica com filtro de score mínimo.

        Converte a distância coseno do ChromaDB em similaridade:
        ``similarity = 1 - distance`` (ChromaDB normaliza distâncias coseno
        para o intervalo [0, 1] quando ``hnsw:space`` é ``cosine``).

        Args:
            query: Texto da consulta.
            top_k: Número máximo de resultados a retornar antes do filtro.
            min_score: Score mínimo de similaridade (0.0–1.0). Documentos
                       abaixo deste limiar são descartados (Requirement 3.2).

        Returns:
            Lista de RetrievedDocument ordenada por score decrescente,
            contendo apenas documentos com score >= min_score.
        """
        if not query or not query.strip():
            return []

        current_count = self._collection.count()
        if current_count == 0:
            return []

        n_results = min(top_k, current_count)

        query_embedding = self._model.encode(query).tolist()

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("Erro ao consultar a base vetorial: %s", exc)
            return []

        retrieved: list[RetrievedDocument] = []

        ids_list = results.get("ids", [[]])[0]
        distances_list = results.get("distances", [[]])[0]
        metadatas_list = results.get("metadatas", [[]])[0]
        documents_list = results.get("documents", [[]])[0]

        for doc_id, distance, metadata, content in zip(
            ids_list, distances_list, metadatas_list, documents_list
        ):
            # ChromaDB com hnsw:space=cosine retorna distâncias em [0, 1]
            # onde 0 = idêntico e 1 = completamente diferente.
            similarity = 1.0 - distance

            if similarity < min_score:
                continue

            document = Document(
                content=content,
                content_hash=doc_id,  # o ID é o content_hash
                source_url=metadata.get("source_url", ""),
                line_name=metadata.get("line_name", ""),
                line_number=metadata.get("line_number", ""),
                chunk_index=int(metadata.get("chunk_index", 0)),
            )
            retrieved.append(RetrievedDocument(document=document, similarity_score=similarity))

        # Garante ordenação decrescente por score
        retrieved.sort(key=lambda r: r.similarity_score, reverse=True)
        return retrieved

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Retorna o número de documentos na coleção."""
        return self._collection.count()

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _get_existing_ids(self) -> set[str]:
        """Retorna o conjunto de IDs (content_hash) já presentes na coleção."""
        if self._collection.count() == 0:
            return set()
        result = self._collection.get(include=[])
        return set(result.get("ids", []))

    def _prepare_batch(
        self,
        documents: list[Document],
    ) -> tuple[list[str], list[list[float]], list[dict], list[str]]:
        """Prepara os dados para inserção em lote no ChromaDB.

        Args:
            documents: Documentos a preparar.

        Returns:
            Tupla (ids, embeddings, metadatas, contents).
        """
        contents = [doc.content for doc in documents]
        embeddings = self._model.encode(contents).tolist()

        ids = [doc.content_hash for doc in documents]
        metadatas = [
            {
                "source_url": doc.source_url,
                "line_name": doc.line_name,
                "line_number": doc.line_number,
                "chunk_index": doc.chunk_index,
            }
            for doc in documents
        ]

        return ids, embeddings, metadatas, contents
