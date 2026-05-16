"""LLM Response Generator — geração de respostas em linguagem natural.

Suporta API OpenAI (gpt-4o-mini) e Ollama via variável de ambiente
LLM_PROVIDER. Responde em português brasileiro com no máximo 150 palavras.
"""

import logging
import os
from typing import Optional

from movebus.models import GenerationResult

logger = logging.getLogger(__name__)

_MAX_WORDS = 150

_SYSTEM_PROMPT = (
    "Você é um assistente de transporte público de Joinville/SC. "
    "Responda APENAS com base nas informações fornecidas no contexto abaixo. "
    "Se o contexto não contiver informações suficientes para responder, "
    "informe que não possui dados sobre o assunto e sugira consultar o site onibus.info. "
    f"Responda em português brasileiro, de forma clara e objetiva, com no máximo {_MAX_WORDS} palavras."
)

_FALLBACK_MESSAGE = (
    "Desculpe, não encontrei informações sobre isso na minha base de dados. "
    "Para consultar horários e itinerários atualizados, acesse diretamente o site onibus.info."
)


def _count_words(text: str) -> int:
    return len(text.split())


class ResponseGenerator:
    """Gera respostas em linguagem natural usando um LLM.

    Suporta dois provedores configuráveis via variável de ambiente
    ``LLM_PROVIDER``:
    - ``openai`` (padrão): usa a API OpenAI com o modelo ``gpt-4o-mini``.
    - ``ollama``: usa Ollama local com o modelo configurado em
      ``OLLAMA_MODEL`` (padrão: ``llama3``).
    """

    def __init__(self) -> None:
        self._provider = os.environ.get("LLM_PROVIDER", "openai").lower()
        self._client: Optional[object] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, question: str, context: str) -> GenerationResult:
        """Gera resposta em linguagem natural baseada no contexto.

        Quando o contexto está vazio, retorna mensagem de fallback sem
        chamar o LLM.

        Args:
            question: Pergunta do usuário (texto transcrito).
            context: Contexto consolidado do RAG.

        Returns:
            GenerationResult com texto da resposta, flag is_fallback e
            contagem de palavras.
        """
        if not context or not context.strip():
            logger.info("Contexto vazio — retornando resposta de fallback.")
            return GenerationResult(
                text=_FALLBACK_MESSAGE,
                is_fallback=True,
                word_count=_count_words(_FALLBACK_MESSAGE),
            )

        try:
            text = self._call_llm(question, context)
        except Exception as exc:
            logger.error("Erro ao chamar o LLM: %s", exc)
            return GenerationResult(
                text=_FALLBACK_MESSAGE,
                is_fallback=True,
                word_count=_count_words(_FALLBACK_MESSAGE),
            )

        word_count = _count_words(text)
        logger.info("Resposta gerada: %d palavras.", word_count)
        return GenerationResult(text=text, is_fallback=False, word_count=word_count)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, question: str, context: str) -> str:
        """Chama o LLM configurado e retorna o texto da resposta."""
        if self._provider == "ollama":
            return self._call_ollama(question, context)
        return self._call_openai(question, context)

    def _call_openai(self, question: str, context: str) -> str:
        """Chama a API OpenAI."""
        from openai import OpenAI  # importação lazy para não exigir a lib se usar Ollama

        if self._client is None:
            self._client = OpenAI()

        user_message = f"Contexto:\n{context}\n\nPergunta: {question}"

        response = self._client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _call_ollama(self, question: str, context: str) -> str:
        """Chama o Ollama local."""
        try:
            from langchain_ollama import OllamaLLM
        except ImportError:
            from langchain_community.llms import Ollama as OllamaLLM

        model_name = os.environ.get("OLLAMA_MODEL", "llama3")
        if self._client is None:
            self._client = OllamaLLM(model=model_name)

        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Contexto:\n{context}\n\n"
            f"Pergunta: {question}"
        )
        return self._client.invoke(prompt).strip()
