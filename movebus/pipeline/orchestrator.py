"""Pipeline Orchestrator — orquestração do pipeline STT → RAG → LLM → TTS.

Executa as etapas sequencialmente, trata erros por etapa e registra
métricas de tempo de execução.
"""

import logging
import tempfile
import time
from pathlib import Path

from movebus.llm.generator import ResponseGenerator
from movebus.models import PipelineResult
from movebus.rag.retriever import BusInfoRetriever
from movebus.stt.transcriber import SpeechTranscriber
from movebus.tts.synthesizer import SpeechSynthesizer

logger = logging.getLogger(__name__)

_ERROR_MESSAGE = (
    "Desculpe, não foi possível processar sua solicitação. Por favor, tente novamente."
)


class QueryPipeline:
    """Orquestra o pipeline completo de consulta por voz.

    Executa STT → RAG → LLM → TTS de forma síncrona. Em caso de falha
    em qualquer etapa, gera áudio de erro via TTS e retorna resultado
    com success=False.
    """

    def __init__(
        self,
        transcriber: SpeechTranscriber,
        retriever: BusInfoRetriever,
        generator: ResponseGenerator,
        synthesizer: SpeechSynthesizer,
        output_dir: str = "./output",
    ) -> None:
        """Inicializa o pipeline com os componentes necessários.

        Args:
            transcriber: Componente STT.
            retriever: Componente RAG.
            generator: Componente LLM.
            synthesizer: Componente TTS.
            output_dir: Diretório para salvar os arquivos de áudio de resposta.
        """
        self._transcriber = transcriber
        self._retriever = retriever
        self._generator = generator
        self._synthesizer = synthesizer
        self._output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def process(self, audio_path: str) -> PipelineResult:
        """Executa o pipeline completo para um arquivo de áudio de entrada.

        Args:
            audio_path: Caminho para o arquivo de áudio da pergunta.

        Returns:
            PipelineResult com texto transcrito, resposta, caminho do áudio
            de resposta, flag de sucesso e tempo total de execução.
        """
        start = time.monotonic()
        logger.info("Pipeline iniciado para '%s'.", audio_path)

        # --- Etapa 1: STT ---
        try:
            transcription = self._transcriber.transcribe(audio_path)
            if not transcription.is_valid:
                error_msg = transcription.error or "Transcrição inválida"
                logger.error("STT falhou: %s", error_msg)
                return self._error_result("stt", error_msg, start)
        except Exception as exc:
            logger.exception("Exceção na etapa STT.")
            return self._error_result("stt", str(exc), start)

        question = transcription.text
        logger.info("STT concluído: '%s'", question)

        # --- Etapa 2: RAG ---
        try:
            retrieval = self._retriever.retrieve(question)
        except Exception as exc:
            logger.exception("Exceção na etapa RAG.")
            return self._error_result("rag", str(exc), start, transcribed_text=question)

        logger.info(
            "RAG concluído: has_results=%s, docs=%d",
            retrieval.has_results,
            len(retrieval.documents),
        )

        # --- Etapa 3: LLM ---
        try:
            generation = self._generator.generate(question, retrieval.context)
        except Exception as exc:
            logger.exception("Exceção na etapa LLM.")
            return self._error_result("llm", str(exc), start, transcribed_text=question)

        response_text = generation.text
        logger.info("LLM concluído: %d palavras.", generation.word_count)

        # --- Etapa 4: TTS ---
        output_path = self._make_output_path()
        try:
            synthesis = self._synthesizer.synthesize(response_text, output_path)
            if synthesis.error:
                logger.error("TTS falhou: %s", synthesis.error)
                return self._error_result(
                    "tts", synthesis.error, start, transcribed_text=question
                )
        except Exception as exc:
            logger.exception("Exceção na etapa TTS.")
            return self._error_result("tts", str(exc), start, transcribed_text=question)

        elapsed = time.monotonic() - start
        logger.info(
            "Pipeline concluído com sucesso em %.2fs. Etapas: STT → RAG → LLM → TTS.",
            elapsed,
        )

        return PipelineResult(
            transcribed_text=question,
            response_text=response_text,
            response_audio_path=synthesis.audio_path,
            success=True,
            error_stage=None,
            total_time_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _error_result(
        self,
        stage: str,
        error_msg: str,
        start: float,
        transcribed_text: str = "",
    ) -> PipelineResult:
        """Gera resultado de erro com áudio de mensagem de erro."""
        error_audio_path = self._make_output_path(prefix="error_")
        try:
            self._synthesizer.synthesize(_ERROR_MESSAGE, error_audio_path)
        except Exception:
            logger.exception("Falha ao gerar áudio de erro.")
            error_audio_path = ""

        elapsed = time.monotonic() - start
        return PipelineResult(
            transcribed_text=transcribed_text,
            response_text=_ERROR_MESSAGE,
            response_audio_path=error_audio_path,
            success=False,
            error_stage=stage,
            total_time_seconds=elapsed,
        )

    def _make_output_path(self, prefix: str = "response_") -> str:
        """Gera um caminho único para o arquivo de saída."""
        with tempfile.NamedTemporaryFile(
            dir=self._output_dir,
            prefix=prefix,
            suffix=".mp3",
            delete=False,
        ) as tmp:
            return tmp.name
