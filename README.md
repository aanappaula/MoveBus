# MoveBus 🚌

Assistente de voz com IA para consulta de transporte público em Joinville/SC.

Você faz uma pergunta por áudio — o sistema transcreve, busca as informações relevantes nas linhas de ônibus e responde em voz.

---

## Como funciona

O pipeline segue quatro etapas em sequência:

```
Áudio → STT → RAG → LLM → TTS → Áudio de resposta
```

1. **STT** — O áudio é transcrito para texto usando o modelo Whisper (OpenAI)
2. **RAG** — O texto é usado para buscar os documentos mais relevantes na base vetorial (ChromaDB + sentence-transformers)
3. **LLM** — O modelo de linguagem gera uma resposta em português com base nos documentos encontrados
4. **TTS** — A resposta é sintetizada em áudio usando gTTS (Google TTS) com fallback para edge-tts

A base de conhecimento é populada com dados do site [onibus.info](https://onibus.info/joinville/) via scraping com Playwright.

---

## Requisitos

- Python 3.12+
- [ffmpeg](https://ffmpeg.org/) instalado e no PATH
- [Ollama](https://ollama.com/) (para LLM local) ou chave da API OpenAI
- Microsoft C++ Build Tools (para compilar o `chroma-hnswlib`)

---

## Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/movebus.git
cd movebus

# 2. Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Instale as dependências
pip install setuptools
pip install -r requirements.txt --no-build-isolation

# 4. Instale o browser do Playwright
python -m playwright install chromium
```

---

## Configuração

Defina as variáveis de ambiente conforme o provedor de LLM desejado:

**Ollama (local):**
```bash
$env:LLM_PROVIDER = "ollama"
$env:OLLAMA_MODEL = "phi3:mini"   # ou llama3, mistral, etc.
```

**OpenAI:**
```bash
$env:LLM_PROVIDER = "openai"
$env:OPENAI_API_KEY = "sk-..."
```

---

## Uso

### 1. Indexar os dados

Faz o scraping do onibus.info e popula a base vetorial. Demora ~15 minutos na primeira vez.

```bash
python -m movebus.cli --update-index
```

### 2. Interface web

```bash
uvicorn movebus.api.app:app --host 0.0.0.0 --port 8000
```

Acesse `http://localhost:8000` — grave sua pergunta pelo microfone ou faça upload de um arquivo de áudio (WAV, MP3, OGG).

### 3. CLI

```bash
python -m movebus.cli --input pergunta.ogg
python -m movebus.cli --input pergunta.ogg --output resposta.mp3
```

---

## Estrutura do projeto

```
movebus/
├── indexer/
│   ├── scraper.py       # Scraping do onibus.info via Playwright
│   ├── chunker.py       # Segmentação de texto em chunks (tiktoken)
│   ├── vector_store.py  # ChromaDB + embeddings multilinguais
│   └── indexer.py       # Orquestração do pipeline de indexação
├── stt/
│   └── transcriber.py   # Speech-to-Text com Whisper
├── rag/
│   └── retriever.py     # Busca semântica na base vetorial
├── llm/
│   └── generator.py     # Geração de resposta (OpenAI / Ollama)
├── tts/
│   └── synthesizer.py   # Text-to-Speech (gTTS + edge-tts)
├── pipeline/
│   └── orchestrator.py  # Orquestração STT → RAG → LLM → TTS
├── api/
│   └── app.py           # Interface web (FastAPI)
├── cli.py               # Interface de linha de comando
└── models.py            # Dataclasses compartilhados
```

---

## Tecnologias

| Componente | Tecnologia |
|---|---|
| STT | [openai-whisper](https://github.com/openai/whisper) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) — `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector store | [ChromaDB](https://www.trychroma.com/) |
| LLM | [Ollama](https://ollama.com/) / [OpenAI](https://openai.com/) via LangChain |
| TTS | [gTTS](https://gtts.readthedocs.io/) + [edge-tts](https://github.com/rany2/edge-tts) |
| Scraping | [Playwright](https://playwright.dev/python/) |
| API | [FastAPI](https://fastapi.tiangolo.com/) |

---

## Equipe

Projeto desenvolvido na disciplina **Fundamentos de Inteligência Artificial** da **Universidade da Região de Joinville (Univille)**, nos cursos de Engenharia de Software e Sistemas de Informação.

**Professor:** Laurindo

**Alunas:**
- Ana Paula
- Ellen Beatriz
- Milena Damasia
- Thaynara Peron
