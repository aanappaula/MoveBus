"""Chunker para segmentação de texto em chunks de tamanho controlado.

Utiliza tiktoken para contagem de tokens e divisão por sentenças com
sobreposição de 50 tokens para preservar contexto entre chunks.
"""

import re
import tiktoken


class TextChunker:
    """Segmenta texto em chunks de no máximo max_tokens tokens.

    Estratégia: divisão por sentenças com sobreposição de 50 tokens entre
    chunks consecutivos, garantindo que o contexto não seja perdido nas
    fronteiras dos chunks.
    """

    OVERLAP_TOKENS = 50

    def __init__(self, encoding_name: str = "cl100k_base"):
        """Inicializa o chunker com o encoding tiktoken especificado.

        Args:
            encoding_name: Nome do encoding tiktoken a usar.
                           "cl100k_base" é compatível com GPT-3.5/GPT-4.
        """
        self._encoding = tiktoken.get_encoding(encoding_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[int]:
        """Retorna a lista de tokens para o texto fornecido."""
        return self._encoding.encode(text)

    def _token_count(self, text: str) -> int:
        """Retorna o número de tokens do texto."""
        return len(self._tokenize(text))

    def _decode(self, tokens: list[int]) -> str:
        """Decodifica uma lista de tokens de volta para texto."""
        return self._encoding.decode(tokens)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Divide o texto em sentenças usando pontuação como delimitador.

        Mantém o delimitador junto à sentença anterior para preservar a
        pontuação original no texto reconstruído.
        """
        # Divide em pontos finais, exclamação, interrogação e quebras de linha
        parts = re.split(r"(?<=[.!?\n])\s+", text.strip())
        # Filtra partes vazias que possam surgir de múltiplos espaços/quebras
        return [p for p in parts if p.strip()]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str, max_tokens: int = 512) -> list[str]:
        """Segmenta texto em chunks de no máximo max_tokens tokens.

        Args:
            text: Texto a ser segmentado.
            max_tokens: Número máximo de tokens por chunk (padrão: 512).

        Returns:
            Lista de strings, cada uma com no máximo max_tokens tokens.
            Retorna lista vazia se o texto for vazio ou contiver apenas
            espaços em branco.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)

        # Caso especial: texto sem pontuação (uma única "sentença")
        # Pode ainda exceder max_tokens — tratamos abaixo.
        chunks: list[str] = []
        overlap_tokens: int = min(self.OVERLAP_TOKENS, max_tokens // 2)

        current_tokens: list[int] = []

        for sentence in sentences:
            sentence_tokens = self._tokenize(sentence)

            # Se a sentença sozinha excede max_tokens, divide-a em sub-chunks
            if len(sentence_tokens) > max_tokens:
                # Primeiro, fecha o chunk atual se houver conteúdo
                if current_tokens:
                    chunks.append(self._decode(current_tokens))
                    current_tokens = []

                # Divide a sentença longa em pedaços de max_tokens com overlap
                start = 0
                while start < len(sentence_tokens):
                    end = start + max_tokens
                    sub_chunk_tokens = sentence_tokens[start:end]
                    chunks.append(self._decode(sub_chunk_tokens))
                    # Avança com overlap
                    start = end - overlap_tokens
                    if start >= len(sentence_tokens):
                        break
                continue

            # Verifica se adicionar a sentença ao chunk atual excederia o limite
            if len(current_tokens) + len(sentence_tokens) > max_tokens:
                # Fecha o chunk atual
                if current_tokens:
                    chunks.append(self._decode(current_tokens))
                    # Inicia o próximo chunk com os últimos overlap_tokens do chunk fechado
                    current_tokens = current_tokens[-overlap_tokens:] if overlap_tokens > 0 else []

            # Adiciona a sentença ao chunk atual
            # Adiciona espaço separador se o chunk já tem conteúdo
            if current_tokens:
                space_tokens = self._tokenize(" ")
                # Verifica se ainda cabe com o espaço
                if len(current_tokens) + len(space_tokens) + len(sentence_tokens) <= max_tokens:
                    current_tokens = current_tokens + space_tokens + sentence_tokens
                else:
                    # Fecha e inicia novo chunk
                    chunks.append(self._decode(current_tokens))
                    current_tokens = current_tokens[-overlap_tokens:] if overlap_tokens > 0 else []
                    current_tokens = current_tokens + sentence_tokens
            else:
                current_tokens = sentence_tokens

        # Adiciona o último chunk se houver conteúdo restante
        if current_tokens:
            chunks.append(self._decode(current_tokens))

        return chunks
