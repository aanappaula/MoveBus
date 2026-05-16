"""Módulo STT — Speech-to-Text do MoveBus.

Transcreve arquivos de áudio para texto usando o modelo Whisper da OpenAI.
Suporta os formatos WAV, MP3 e OGG, com timeout de 10 segundos.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

import whisper

from movebus.models import TranscriptionResult

logger = logging.getLogger(__name__)

# Formatos de áudio aceitos (validação por extensão)
SUPPORTED_FORMATS = {".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"}

# Timeout em segundos para a transcrição
TRANSCRIPTION_TIMEOUT = 60

# Tamanho mínimo de texto para ser considerado válido
MIN_TEXT_LENGTH = 3


def _get_model_size() -> str:
    """Retorna o tamanho do modelo Whisper configurado via variável de ambiente."""
    return os.environ.get("WHISPER_MODEL_SIZE", "base")


class SpeechTranscriber:
    """Transcreve arquivos de áudio para texto usando o modelo Whisper.

    O modelo é carregado na primeira chamada a `transcribe` e reutilizado
    nas chamadas subsequentes (lazy loading).
    """

    def __init__(self) -> None:
        self._model = None
        self._model_size: str | None = None

    def _load_model(self) -> whisper.Whisper:
        """Carrega (ou reutiliza) o modelo Whisper configurado."""
        model_size = _get_model_size()
        if self._model is None or self._model_size != model_size:
            logger.info("Carregando modelo Whisper '%s'...", model_size)
            self._model = whisper.load_model(model_size)
            self._model_size = model_size
        return self._model

    def _run_transcription(self, audio_path: str) -> str:
        """Executa a transcrição de forma síncrona (chamado dentro do executor)."""
        model = self._load_model()
        result = model.transcribe(audio_path, language="pt", fp16=False)
        return result.get("text", "").strip()

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcreve arquivo de áudio para texto.

        Valida o formato do arquivo antes de chamar o modelo. A transcrição
        é executada com timeout de 10 segundos via ThreadPoolExecutor.

        Args:
            audio_path: Caminho para o arquivo de áudio (WAV, MP3 ou OGG).

        Returns:
            TranscriptionResult com campos:
              - text: str (texto transcrito)
              - is_valid: bool (False se texto < 3 caracteres)
              - error: str | None (mensagem de erro se falhou)
        """
        # --- Validação de formato ---
        path = Path(audio_path)
        extension = path.suffix.lower()
        converted_path = None

        if extension not in SUPPORTED_FORMATS:
            logger.warning(
                "Formato de áudio não suportado: '%s' (arquivo: %s)",
                extension,
                audio_path,
            )
            return TranscriptionResult(
                text="",
                is_valid=False,
                error="Formato não suportado. Aceitos: WAV, MP3, OGG",
            )

        # --- Conversão de formato se necessário ---
        # Whisper suporta webm/m4a via ffmpeg, mas converte explicitamente
        # para WAV para garantir compatibilidade
        converted_path = None
        if extension in {".webm", ".m4a", ".flac"}:
            try:
                import subprocess
                import tempfile as _tempfile
                tmp = _tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                converted_path = tmp.name
                tmp.close()
                subprocess.run(
                    ["ffmpeg", "-y", "-i", audio_path, converted_path],
                    capture_output=True, check=True
                )
                audio_path = converted_path
            except Exception as exc:
                logger.error("Erro ao converter áudio: %s", exc)
                return TranscriptionResult(
                    text="", is_valid=False,
                    error="Erro ao converter formato de áudio",
                )

        # --- Transcrição com timeout ---
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._run_transcription, audio_path)
                text = future.result(timeout=TRANSCRIPTION_TIMEOUT)

        except FuturesTimeoutError:
            logger.error(
                "Timeout ao transcrever '%s' (limite: %ds)",
                audio_path,
                TRANSCRIPTION_TIMEOUT,
            )
            if converted_path:
                Path(converted_path).unlink(missing_ok=True)
            return TranscriptionResult(
                text="",
                is_valid=False,
                error="Tempo limite de transcrição excedido",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Erro ao transcrever '%s': %s", audio_path, exc)
            if converted_path:
                Path(converted_path).unlink(missing_ok=True)
            return TranscriptionResult(
                text="",
                is_valid=False,
                error="Arquivo de áudio corrompido ou ilegível",
            )

        if converted_path:
            Path(converted_path).unlink(missing_ok=True)

        # --- Validação do resultado ---
        is_valid = len(text) >= MIN_TEXT_LENGTH
        logger.info(
            "Transcrição concluída: %d caracteres, válida=%s",
            len(text),
            is_valid,
        )
        return TranscriptionResult(text=text, is_valid=is_valid, error=None)
