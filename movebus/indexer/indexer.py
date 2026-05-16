"""BusIndexer — orquestração do pipeline de indexação.

Coordena Scraper → Chunker → VectorStore para indexação completa das
linhas de ônibus de Joinville. Preserva a base vetorial existente se o
site estiver indisponível durante a atualização.
"""

import hashlib
import logging
import time

from movebus.indexer.chunker import TextChunker
from movebus.indexer.scraper import BusScraper
from movebus.indexer.vector_store import VectorStore
from movebus.models import BusLineRaw, Document, IndexingResult

logger = logging.getLogger(__name__)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class BusIndexer:
    """Orquestra o pipeline de indexação: Scraper → Chunker → VectorStore.

    Suporta indexação incremental (add_documents) e reindexação completa
    (clear_and_reindex). Preserva a base vetorial existente se o scraping
    não retornar nenhuma linha (site indisponível).
    """

    def __init__(
        self,
        scraper: BusScraper | None = None,
        chunker: TextChunker | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Inicializa o indexador com os componentes necessários.

        Args:
            scraper: Instância do BusScraper. Cria uma nova se None.
            chunker: Instância do TextChunker. Cria uma nova se None.
            vector_store: Instância do VectorStore. Cria uma nova se None.
        """
        self._scraper = scraper or BusScraper()
        self._chunker = chunker or TextChunker()
        self._vector_store = vector_store or VectorStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_reindex(self) -> IndexingResult:
        """Executa reindexação completa: limpa a base e reindexia tudo.

        Se o scraping não retornar nenhuma linha (site indisponível),
        preserva a base vetorial existente e registra o erro em log.

        Returns:
            IndexingResult com estatísticas da operação.
        """
        start = time.monotonic()
        logger.info("Iniciando reindexação completa...")

        raw_lines = self._scraper.scrape_all_lines()

        if not raw_lines:
            elapsed = time.monotonic() - start
            logger.error(
                "Scraping não retornou nenhuma linha. "
                "Base vetorial preservada com dados anteriores."
            )
            return IndexingResult(
                documents_indexed=0,
                documents_removed=0,
                execution_time_seconds=elapsed,
                errors=["Site indisponível ou sem linhas encontradas."],
            )

        documents, scraping_errors = self._build_documents(raw_lines)
        result = self._vector_store.clear_and_reindex(documents)

        # Incorpora erros de scraping ao resultado
        result.errors.extend(scraping_errors)
        result.execution_time_seconds = time.monotonic() - start

        logger.info(
            "Reindexação concluída: %d documentos indexados, %d removidos, "
            "%d erros, %.2fs.",
            result.documents_indexed,
            result.documents_removed,
            len(result.errors),
            result.execution_time_seconds,
        )
        return result

    def run_incremental_index(self) -> IndexingResult:
        """Executa indexação incremental: adiciona apenas documentos novos.

        Documentos já existentes na base (mesmo hash) são ignorados.

        Returns:
            IndexingResult com estatísticas da operação.
        """
        start = time.monotonic()
        logger.info("Iniciando indexação incremental...")

        raw_lines = self._scraper.scrape_all_lines()

        if not raw_lines:
            elapsed = time.monotonic() - start
            logger.error("Scraping não retornou nenhuma linha.")
            return IndexingResult(
                documents_indexed=0,
                documents_removed=0,
                execution_time_seconds=elapsed,
                errors=["Site indisponível ou sem linhas encontradas."],
            )

        documents, scraping_errors = self._build_documents(raw_lines)
        count_before = self._vector_store.count()
        self._vector_store.add_documents(documents)
        count_after = self._vector_store.count()

        elapsed = time.monotonic() - start
        indexed = count_after - count_before

        logger.info(
            "Indexação incremental concluída: %d novos documentos, "
            "%d erros, %.2fs.",
            indexed,
            len(scraping_errors),
            elapsed,
        )
        return IndexingResult(
            documents_indexed=indexed,
            documents_removed=0,
            execution_time_seconds=elapsed,
            errors=scraping_errors,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_documents(
        self, raw_lines: list[BusLineRaw]
    ) -> tuple[list[Document], list[str]]:
        """Converte linhas brutas em Documents prontos para indexação.

        Args:
            raw_lines: Lista de BusLineRaw coletados pelo scraper.

        Returns:
            Tupla (documents, errors) onde errors são URLs que falharam.
        """
        documents: list[Document] = []
        errors: list[str] = []

        for line in raw_lines:
            try:
                text = self._build_line_text(line)
                chunks = self._chunker.chunk(text)

                for idx, chunk in enumerate(chunks):
                    content_hash = _sha256(chunk)
                    documents.append(
                        Document(
                            content=chunk,
                            content_hash=content_hash,
                            source_url=line.url,
                            line_name=line.line_name,
                            line_number=line.line_number,
                            chunk_index=idx,
                        )
                    )
            except Exception:
                logger.exception(
                    "Erro ao processar linha '%s' (%s).",
                    line.line_name,
                    line.url,
                )
                errors.append(line.url)

        logger.info(
            "%d documentos gerados a partir de %d linhas (%d erros).",
            len(documents),
            len(raw_lines),
            len(errors),
        )
        return documents, errors

    @staticmethod
    def _build_line_text(line: BusLineRaw) -> str:
        """Constrói o texto completo de uma linha para indexação."""
        parts = [
            f"Linha {line.line_number} — {line.line_name}",
            f"Itinerário: {line.itinerary}" if line.itinerary else "",
            "Horários: " + ", ".join(line.schedules) if line.schedules else "",
            "Pontos de parada: " + " | ".join(line.stops) if line.stops else "",
        ]
        return "\n".join(p for p in parts if p)
