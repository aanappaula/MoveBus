"""TTS Synthesizer — síntese de voz em português brasileiro.

Implementação primária com gTTS (Google TTS) e fallback para edge-tts
(Microsoft Edge TTS) em caso de falha. Suporta saída em MP3 e WAV.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path

from movebus.models import SynthesisResult

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = {"mp3", "wav"}


class SpeechSynthesizer:
    """Sintetiza texto em áudio usando gTTS com fallback para edge-tts.

    Implementação primária: gTTS com lang='pt' e tld='com.br'.
    Fallback: edge-tts com voz 'pt-BR-FranciscaNeural'.
    Conversão para WAV via pydub quando o formato solicitado for 'wav'.
    """

    def synthesize(self, text: str, output_path: str) -> SynthesisResult:
        """Sintetiza texto em arquivo de áudio.

        Args:
            text: Texto a ser sintetizado.
            output_path: Caminho do arquivo de saída (extensão define o
                         formato: .mp3 ou .wav).

        Returns:
            SynthesisResult com audio_path, format e error (se houver).
        """
        if not text or not text.strip():
            logger.warning("Texto vazio recebido pelo TTS.")
            return SynthesisResult(
                audio_path="",
                format="",
                error="Nenhum conteúdo para sintetizar",
            )

        output_format = Path(output_path).suffix.lstrip(".").lower() or "mp3"
        if output_format not in _SUPPORTED_FORMATS:
            output_format = "mp3"

        # Garante que o diretório de saída existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Tenta gTTS primeiro
        result = self._synthesize_gtts(text, output_path, output_format)
        if result.error is None:
            return result

        # Fallback para edge-tts
        logger.warning("gTTS falhou (%s). Tentando edge-tts...", result.error)
        return self._synthesize_edge_tts(text, output_path, output_format)

    # ------------------------------------------------------------------
    # gTTS
    # ------------------------------------------------------------------

    def _synthesize_gtts(
        self, text: str, output_path: str, output_format: str
    ) -> SynthesisResult:
        """Sintetiza usando gTTS."""
        try:
            from gtts import gTTS

            tts = gTTS(text=text, lang="pt", tld="com.br", slow=False)

            if output_format == "wav":
                # gTTS só gera MP3; converte para WAV via pydub
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    tts.save(tmp_path)
                    self._convert_mp3_to_wav(tmp_path, output_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            else:
                tts.save(output_path)

            logger.info("gTTS: áudio salvo em '%s'.", output_path)
            return SynthesisResult(audio_path=output_path, format=output_format)

        except Exception as exc:
            logger.error("Erro no gTTS: %s", exc)
            return SynthesisResult(audio_path="", format="", error=str(exc))

    # ------------------------------------------------------------------
    # edge-tts (fallback)
    # ------------------------------------------------------------------

    def _synthesize_edge_tts(
        self, text: str, output_path: str, output_format: str
    ) -> SynthesisResult:
        """Sintetiza usando edge-tts como fallback."""
        try:
            import edge_tts

            voice = "pt-BR-FranciscaNeural"

            if output_format == "wav":
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    asyncio.run(self._edge_tts_save(edge_tts, text, voice, tmp_path))
                    self._convert_mp3_to_wav(tmp_path, output_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            else:
                asyncio.run(self._edge_tts_save(edge_tts, text, voice, output_path))

            logger.info("edge-tts: áudio salvo em '%s'.", output_path)
            return SynthesisResult(audio_path=output_path, format=output_format)

        except Exception as exc:
            logger.error("Erro no edge-tts: %s", exc)
            return SynthesisResult(
                audio_path="",
                format="",
                error=f"Falha em ambos os provedores TTS: {exc}",
            )

    @staticmethod
    async def _edge_tts_save(edge_tts, text: str, voice: str, path: str) -> None:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(path)

    # ------------------------------------------------------------------
    # Conversão de formato
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_mp3_to_wav(mp3_path: str, wav_path: str) -> None:
        """Converte arquivo MP3 para WAV usando pydub."""
        from pydub import AudioSegment

        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
