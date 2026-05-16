"""CLI do MoveBus — interface de linha de comando.

Uso:
    python -m movebus.cli --input pergunta.wav --output resposta.mp3
    python -m movebus.cli --update-index
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _build_pipeline(output_dir: str):
    """Instancia e retorna o QueryPipeline com todos os componentes."""
    from movebus.indexer.vector_store import VectorStore
    from movebus.llm.generator import ResponseGenerator
    from movebus.pipeline.orchestrator import QueryPipeline
    from movebus.rag.retriever import BusInfoRetriever
    from movebus.stt.transcriber import SpeechTranscriber
    from movebus.tts.synthesizer import SpeechSynthesizer

    vector_store = VectorStore()
    return QueryPipeline(
        transcriber=SpeechTranscriber(),
        retriever=BusInfoRetriever(vector_store),
        generator=ResponseGenerator(),
        synthesizer=SpeechSynthesizer(),
        output_dir=output_dir,
    )


def cmd_query(args: argparse.Namespace) -> int:
    """Processa uma pergunta de áudio e retorna a resposta em áudio."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Erro: arquivo não encontrado: {args.input}", file=sys.stderr)
        return 1

    output_dir = str(Path(args.output).parent) if args.output else "./output"
    pipeline = _build_pipeline(output_dir)
    result = pipeline.process(str(input_path))

    print(f"Texto transcrito : {result.transcribed_text}")
    print(f"Resposta         : {result.response_text}")
    print(f"Áudio de resposta: {result.response_audio_path}")

    if not result.success:
        print(
            f"Aviso: pipeline falhou na etapa '{result.error_stage}'.",
            file=sys.stderr,
        )
        return 1

    # Renomeia o arquivo de saída se --output foi especificado
    if args.output and result.response_audio_path:
        import shutil
        shutil.move(result.response_audio_path, args.output)
        print(f"Áudio salvo em   : {args.output}")

    return 0


def cmd_update_index(_args: argparse.Namespace) -> int:
    """Executa reindexação completa da base vetorial."""
    from movebus.indexer.indexer import BusIndexer

    indexer = BusIndexer()
    result = indexer.run_full_reindex()

    print(f"Documentos indexados : {result.documents_indexed}")
    print(f"Documentos removidos : {result.documents_removed}")
    print(f"Tempo de execução    : {result.execution_time_seconds:.2f}s")

    if result.errors:
        print(f"Erros ({len(result.errors)}):", file=sys.stderr)
        for err in result.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="movebus",
        description="Assistente de voz para transporte público em Joinville/SC.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Subcomando: query (padrão quando --input é fornecido diretamente)
    parser.add_argument("--input", metavar="AUDIO", help="Arquivo de áudio de entrada.")
    parser.add_argument(
        "--output", metavar="AUDIO", help="Arquivo de áudio de saída (opcional)."
    )
    parser.add_argument(
        "--update-index",
        action="store_true",
        help="Atualiza a base vetorial com dados do onibus.info.",
    )

    args = parser.parse_args()

    if args.update_index:
        sys.exit(cmd_update_index(args))
    elif args.input:
        sys.exit(cmd_query(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
