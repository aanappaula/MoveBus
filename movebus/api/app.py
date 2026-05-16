"""API HTTP do MoveBus via FastAPI."""

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from movebus.indexer.vector_store import VectorStore
from movebus.llm.generator import ResponseGenerator
from movebus.pipeline.orchestrator import QueryPipeline
from movebus.rag.retriever import BusInfoRetriever
from movebus.stt.transcriber import SpeechTranscriber
from movebus.tts.synthesizer import SpeechSynthesizer

logger = logging.getLogger(__name__)

app = FastAPI(title="MoveBus API", version="1.0.0")

_pipeline: QueryPipeline | None = None


def _get_pipeline() -> QueryPipeline:
    global _pipeline
    if _pipeline is None:
        vector_store = VectorStore()
        _pipeline = QueryPipeline(
            transcriber=SpeechTranscriber(),
            retriever=BusInfoRetriever(vector_store),
            generator=ResponseGenerator(),
            synthesizer=SpeechSynthesizer(),
            output_dir="./output",
        )
    return _pipeline


_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MoveBus</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', sans-serif;
      background: #f0f4f8;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
    }
    .card {
      background: white;
      border-radius: 16px;
      padding: 40px;
      width: 100%;
      max-width: 480px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.10);
      text-align: center;
    }
    .logo { font-size: 2.5rem; margin-bottom: 8px; }
    h1 { font-size: 1.5rem; color: #1a202c; margin-bottom: 4px; }
    .subtitle { color: #718096; font-size: 0.95rem; margin-bottom: 28px; }
    .btn-record {
      width: 88px; height: 88px;
      border-radius: 50%;
      border: none;
      background: #3182ce;
      color: white;
      font-size: 2rem;
      cursor: pointer;
      transition: all 0.2s;
      box-shadow: 0 4px 12px rgba(49,130,206,0.4);
      display: flex; align-items: center; justify-content: center;
      margin: 0 auto 12px;
    }
    .btn-record:hover { background: #2b6cb0; transform: scale(1.05); }
    .btn-record.recording {
      background: #e53e3e;
      animation: pulse 1s infinite;
      box-shadow: 0 4px 12px rgba(229,62,62,0.4);
    }
    @keyframes pulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.08); }
    }
    .status { color: #4a5568; font-size: 0.88rem; min-height: 20px; margin-bottom: 16px; }
    .divider { color: #cbd5e0; font-size: 0.82rem; margin: 4px 0 14px; }
    .btn-upload {
      display: inline-block;
      padding: 9px 22px;
      border: 2px solid #3182ce;
      border-radius: 8px;
      color: #3182ce;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      margin-bottom: 20px;
    }
    .btn-upload:hover { background: #ebf8ff; }
    .spinner {
      display: none;
      width: 30px; height: 30px;
      border: 3px solid #e2e8f0;
      border-top-color: #3182ce;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin: 0 auto 16px;
    }
    .spinner.visible { display: block; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .result { background: #f7fafc; border-radius: 10px; padding: 16px; text-align: left; display: none; }
    .result.visible { display: block; }
    .result-label { font-size: 0.72rem; font-weight: 700; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
    .result-text { color: #2d3748; font-size: 0.93rem; line-height: 1.5; margin-bottom: 12px; }
    audio { width: 100%; border-radius: 8px; }
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">🚌</div>
    <h1>MoveBus</h1>
    <p class="subtitle">Assistente de transporte público de Joinville</p>

    <button class="btn-record" id="btnRecord">🎤</button>
    <div class="status" id="status">Clique no microfone para gravar sua pergunta</div>

    <div class="divider">— ou —</div>

    <label>
      <input type="file" id="fileInput" accept="audio/*" style="display:none">
      <span class="btn-upload">📁 Enviar arquivo de áudio</span>
    </label>

    <div class="spinner" id="spinner"></div>

    <div class="result" id="result">
      <div class="result-label">Você perguntou</div>
      <div class="result-text" id="transcribed"></div>
      <div class="result-label">Resposta</div>
      <div class="result-text" id="responseText"></div>
      <audio id="audioPlayer" controls autoplay></audio>
    </div>
  </div>

  <script>
    const btn = document.getElementById('btnRecord');
    const status = document.getElementById('status');
    const spinner = document.getElementById('spinner');
    const result = document.getElementById('result');
    const transcribed = document.getElementById('transcribed');
    const responseText = document.getElementById('responseText');
    const audioPlayer = document.getElementById('audioPlayer');
    const fileInput = document.getElementById('fileInput');

    let mediaRecorder = null, chunks = [], recording = false;

    btn.addEventListener('click', async () => {
      recording ? stopRecording() : await startRecording();
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) sendAudioBlob(fileInput.files[0], fileInput.files[0].name);
    });

    async function startRecording() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        chunks = [];
        mediaRecorder.ondataavailable = e => chunks.push(e.data);
        mediaRecorder.onstop = () => {
          const blob = new Blob(chunks, { type: 'audio/webm' });
          sendAudioBlob(blob, 'pergunta.webm');
        };
        mediaRecorder.start();
        recording = true;
        btn.classList.add('recording');
        btn.textContent = '⏹';
        status.textContent = 'Gravando... clique para parar';
        result.classList.remove('visible');
      } catch (e) {
        status.textContent = 'Microfone indisponível: ' + e.message;
      }
    }

    function stopRecording() {
      mediaRecorder?.stop();
      mediaRecorder?.stream.getTracks().forEach(t => t.stop());
      recording = false;
      btn.classList.remove('recording');
      btn.textContent = '🎤';
      status.textContent = 'Processando...';
      spinner.classList.add('visible');
    }

    async function sendAudioBlob(blob, filename) {
      status.textContent = 'Processando...';
      spinner.classList.add('visible');
      result.classList.remove('visible');

      const formData = new FormData();
      formData.append('audio_file', blob, filename);

      try {
        const res = await fetch('/query', { method: 'POST', body: formData });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || 'Erro no servidor');
        }
        const txText = res.headers.get('X-Transcribed-Text') || '';
        const respText = res.headers.get('X-Response-Text') || '';
        transcribed.textContent = decodeURIComponent(txText);
        responseText.textContent = decodeURIComponent(respText);
        const audioBlob = await res.blob();
        audioPlayer.src = URL.createObjectURL(audioBlob);
        audioPlayer.play();
        result.classList.add('visible');
        status.textContent = 'Clique no microfone ou envie outro arquivo';
      } catch (e) {
        status.textContent = 'Erro: ' + e.message;
      } finally {
        spinner.classList.remove('visible');
      }
    }
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=_HTML)


@app.post("/query")
async def query(audio_file: UploadFile = File(...)):
    suffix = Path(audio_file.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(await audio_file.read())

    try:
        result = _get_pipeline().process(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    audio_path = result.response_audio_path
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(status_code=500, detail="Áudio de resposta não gerado")

    def _encode(text: str) -> str:
        from urllib.parse import quote
        return quote(text, safe=" ,./!?-")

    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename="response.mp3",
        headers={
            "X-Transcribed-Text": _encode(result.transcribed_text),
            "X-Response-Text": _encode(result.response_text),
        },
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
