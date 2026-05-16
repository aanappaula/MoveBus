"""Modelos de dados do MoveBus.

Todos os dataclasses utilizados pelos componentes do pipeline de consulta
e do pipeline de indexação estão definidos aqui.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BusLineRaw:
    """Dados brutos coletados pelo scraper."""

    url: str
    line_name: str
    line_number: str
    itinerary: str
    schedules: list[str]
    stops: list[str]
    raw_html: str


@dataclass
class Document:
    """Fragmento de texto indexado na base vetorial."""

    content: str          # texto do chunk (≤512 tokens)
    content_hash: str     # SHA-256 do conteúdo (para deduplicação)
    source_url: str       # URL de origem
    line_name: str        # nome da linha de ônibus
    line_number: str      # número da linha
    chunk_index: int      # índice do chunk dentro da linha


@dataclass
class RetrievedDocument:
    """Documento recuperado com score de similaridade."""

    document: Document
    similarity_score: float   # cosine similarity [0.0, 1.0]


@dataclass
class RetrievalResult:
    """Resultado da etapa RAG."""

    documents: list[RetrievedDocument]
    has_results: bool         # False se nenhum doc atingiu score mínimo
    context: str              # contexto consolidado para o LLM


@dataclass
class TranscriptionResult:
    """Resultado da etapa STT."""

    text: str
    is_valid: bool            # False se texto < 3 caracteres
    error: Optional[str] = None


@dataclass
class GenerationResult:
    """Resultado da etapa LLM."""

    text: str
    is_fallback: bool         # True se contexto insuficiente
    word_count: int


@dataclass
class SynthesisResult:
    """Resultado da etapa TTS."""

    audio_path: str
    format: str               # 'wav' ou 'mp3'
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Resultado completo do pipeline de consulta."""

    transcribed_text: str
    response_text: str
    response_audio_path: str
    success: bool
    error_stage: Optional[str] = None   # 'stt' | 'rag' | 'llm' | 'tts'
    total_time_seconds: float = 0.0


@dataclass
class IndexingResult:
    """Resultado da operação de indexação."""

    documents_indexed: int
    documents_removed: int
    execution_time_seconds: float
    errors: list[str] = field(default_factory=list)   # URLs que falharam durante o scraping
