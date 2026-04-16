"""
Headless Windows port of float-speech-to-text.

- Global hotkey toggles recording (default: Ctrl+Shift)
- Tray icon shows state (idle/recording/processing)
- Tray menu: toggle LLM polish, auto-paste, loading dots, quit
- Inline animated spinner typed into the active field (Listening →
  Transcribing → Polishing)
- ASR: onnx-asr (default model: gigaam-v3-e2e-rnnt, Russian)
- LLM polish backends (optional, pluggable via FSTT_LLM_BACKEND):
    claude — `claude` CLI via subscription (stream-json stdin/stdout)
    ollama — local HTTP; streams tokens directly into the field
    api    — Anthropic SDK direct API call
"""
from __future__ import annotations

import os
import sys
import time
import json
import threading
import queue
import logging
from enum import Enum
from typing import Optional

import subprocess
import shutil
import urllib.request
import urllib.error
import numpy as np
import sounddevice as sd
import pyperclip
import keyboard
import pystray
from PIL import Image, ImageDraw
from dotenv import load_dotenv

load_dotenv()

# ---------- Config ----------
def _env_bool(k: str, d: bool) -> bool:
    v = os.environ.get(k)
    return d if v is None else v.strip().lower() in ("1", "true", "yes", "on")

MODEL_NAME        = os.environ.get("FSTT_ONNX_ASR_MODEL", "gigaam-v3-e2e-rnnt")
SAMPLE_RATE       = int(os.environ.get("FSTT_SAMPLE_RATE", "16000"))
HOTKEY            = os.environ.get("FSTT_HOTKEY", "ctrl+shift")
HOTKEY_MIN_HOLD_MS = int(os.environ.get("FSTT_HOTKEY_MIN_HOLD_MS", "30"))
HOTKEY_MAX_HOLD_MS = int(os.environ.get("FSTT_HOTKEY_MAX_HOLD_MS", "2000"))

LLM_ENABLED       = _env_bool("FSTT_LLM_ENABLED", False)
LLM_PROMPT_FILE   = os.environ.get("FSTT_LLM_PROMPT_FILE", "prompt.md")
LLM_TIMEOUT_SEC   = int(os.environ.get("FSTT_LLM_TIMEOUT_SEC", "60"))
LLM_BACKEND       = os.environ.get("FSTT_LLM_BACKEND", "claude")  # claude | ollama | api
API_MODEL         = os.environ.get("FSTT_API_MODEL", "claude-haiku-4-5")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_BIN        = os.environ.get("FSTT_CLAUDE_BIN", "claude")
CLAUDE_MODEL      = os.environ.get("FSTT_CLAUDE_MODEL", "haiku")
CLAUDE_EFFORT     = os.environ.get("FSTT_CLAUDE_EFFORT", "low")
CLAUDE_MAX_TURNS  = int(os.environ.get("FSTT_CLAUDE_MAX_TURNS", "20"))
OLLAMA_URL        = os.environ.get("FSTT_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.environ.get("FSTT_OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_PROMPT_DEFAULT = (
    "Ты редактор расшифровок голосовых сообщений на русском. "
    "Расставь знаки препинания и удали слова-паразиты (э-э, ну, вот, типа, "
    "как бы, короче). СТРОГО сохраняй оригинальные слова и порядок слов — "
    "никаких синонимов и перефразирования. Выведи ТОЛЬКО исправленный "
    "текст, без пояснений, кавычек и разметки."
)
OLLAMA_PROMPT_FILE = os.environ.get("FSTT_OLLAMA_PROMPT_FILE", "")

AUTO_PASTE        = _env_bool("FSTT_AUTO_PASTE", True)
PLACEHOLDER       = _env_bool("FSTT_PLACEHOLDER", True)

logging.basicConfig(
    level=os.environ.get("FSTT_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fstt")


# ---------- State ----------
class Phase(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


# ---------- ASR ----------
class ASR:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self._model is None:
                import onnx_asr
                log.info("Loading ASR model: %s", self.model_name)
                t0 = time.time()
                self._model = onnx_asr.load_model(self.model_name)
                log.info("ASR model loaded in %.1fs", time.time() - t0)
        return self._model

    def transcribe(self, audio: np.ndarray) -> str:
        model = self.load()
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        result = model.recognize(audio)
        if isinstance(result, list):
            result = result[0] if result else ""
        return (result or "").strip()


# ---------- LLM ----------
def _read_prompt_file(path: str) -> Optional[str]:
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def load_prompt() -> str:
    if LLM_BACKEND == "ollama":
        content = _read_prompt_file(OLLAMA_PROMPT_FILE)
        if content is None:
            return OLLAMA_PROMPT_DEFAULT
        return content
    content = _read_prompt_file(LLM_PROMPT_FILE)
    if content is None:
        log.warning("Prompt file not found: %s — skipping LLM", LLM_PROMPT_FILE)
        return ""
    return content


def _claude_available() -> bool:
    return shutil.which(CLAUDE_BIN) is not None


class ClaudeWorker:
    """Long-lived `claude -p` subprocess fed via stream-json stdin/stdout."""

    def __init__(self, system_prompt: str, model: str, max_turns: int, timeout: int):
        self.system_prompt = system_prompt
        self.model = model
        self.max_turns = max_turns
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen] = None
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._turns = 0
        self._lock = threading.Lock()

    def _spawn(self):
        cmd = [
            CLAUDE_BIN, "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--model", self.model,
            "--effort", CLAUDE_EFFORT,
            "--system-prompt", self.system_prompt,
            "--no-session-persistence",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._q = queue.Queue()
        self._turns = 0
        threading.Thread(target=self._reader, args=(self._proc.stdout, self._q), daemon=True).start()
        log.info("claude worker spawned pid=%s", self._proc.pid)

    @staticmethod
    def _reader(stdout, q: "queue.Queue[Optional[str]]"):
        try:
            for line in stdout:
                q.put(line)
        finally:
            q.put(None)

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _ensure(self):
        if not self._alive():
            self._spawn()
        elif self._turns >= self.max_turns:
            log.info("rotating claude worker after %d turns", self._turns)
            self.close()
            self._spawn()

    def process(self, text: str) -> Optional[str]:
        with self._lock:
            self._ensure()
            msg = json.dumps({
                "type": "user",
                "message": {"role": "user", "content": text},
            }, ensure_ascii=False)
            try:
                self._proc.stdin.write(msg + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._spawn()
                self._proc.stdin.write(msg + "\n")
                self._proc.stdin.flush()

            deadline = time.time() + self.timeout
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    log.error("claude worker timeout after %ds — killing", self.timeout)
                    self.close()
                    return None
                try:
                    line = self._q.get(timeout=remaining)
                except queue.Empty:
                    self.close()
                    return None
                if line is None:
                    log.error("claude worker exited unexpectedly")
                    self._proc = None
                    return None
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if evt.get("type") == "result":
                    self._turns += 1
                    if evt.get("is_error"):
                        log.error("claude result error: %s", evt.get("result"))
                        return None
                    return (evt.get("result") or "").strip()

    def close(self):
        if self._proc is not None:
            try: self._proc.stdin.close()
            except Exception: pass
            try: self._proc.terminate()
            except Exception: pass
            try: self._proc.wait(timeout=2)
            except Exception:
                try: self._proc.kill()
                except Exception: pass
            self._proc = None


_worker: Optional[ClaudeWorker] = None
_worker_lock = threading.Lock()


def _get_worker(system_prompt: str) -> Optional[ClaudeWorker]:
    global _worker
    if not _claude_available():
        return None
    with _worker_lock:
        if _worker is None:
            _worker = ClaudeWorker(
                system_prompt=system_prompt,
                model=CLAUDE_MODEL,
                max_turns=CLAUDE_MAX_TURNS,
                timeout=LLM_TIMEOUT_SEC,
            )
    return _worker


def _ollama_available() -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/version", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


def ollama_stream(text: str, system_prompt: str, on_chunk) -> Optional[str]:
    """Stream tokens from Ollama. Calls on_chunk(str) for each piece.
    Returns the full accumulated text on success, or None on error."""
    if not _ollama_available():
        log.warning("ollama not reachable at %s", OLLAMA_URL)
        return None
    payload = {
        "model": OLLAMA_MODEL,
        "stream": True,
        "options": {"temperature": 0.3},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    }
    req = urllib.request.Request(
        OLLAMA_URL + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    parts = []
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_SEC) as r:
            for raw in r:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                chunk = obj.get("message", {}).get("content", "")
                if chunk:
                    parts.append(chunk)
                    try:
                        on_chunk(chunk)
                    except Exception as e:
                        log.error("on_chunk failed: %s", e)
                if obj.get("done"):
                    break
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.error("ollama stream failed: %s", e)
        return None
    return "".join(parts).strip()


def ollama_process(text: str, system_prompt: str) -> Optional[str]:
    if not _ollama_available():
        log.warning("ollama not reachable at %s", OLLAMA_URL)
        return None
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {"temperature": 0.3},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Отредактируй текст внутри тегов <text> согласно правилам. "
                    "Выведи ТОЛЬКО итоговый отредактированный текст, без пояснений, "
                    "комментариев, приветствий или markdown.\n"
                    f"<text>\n{text}\n</text>"
                ),
            },
        ],
    }
    req = urllib.request.Request(
        OLLAMA_URL + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_SEC) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.error("ollama request failed: %s", e)
        return None
    out = (data.get("message", {}).get("content") or "").strip()
    # strip <text> wrappers if model echoed them
    for tag in ("<text>", "</text>"):
        out = out.replace(tag, "")
    return out.strip()


_api_client = None


def _get_api_client():
    global _api_client
    if _api_client is None and ANTHROPIC_API_KEY:
        import anthropic
        _api_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _api_client


def _api_available() -> bool:
    return bool(ANTHROPIC_API_KEY)


def api_process(text: str, system_prompt: str) -> Optional[str]:
    client = _get_api_client()
    if client is None:
        log.warning("ANTHROPIC_API_KEY not set — skipping API backend")
        return None
    user_msg = (
        "Отредактируй текст внутри тегов <text> согласно правилам из системного "
        "промпта. Выведи ТОЛЬКО итоговый отредактированный текст, без пояснений, "
        "комментариев, приветствий или markdown-форматирования.\n"
        f"<text>\n{text}\n</text>"
    )
    try:
        resp = client.messages.create(
            model=API_MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        log.error("api request failed: %s", e)
        return None
    out = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
    return (out or "").strip()


def claude_process(text: str, system_prompt: str) -> Optional[str]:
    worker = _get_worker(system_prompt)
    if worker is None:
        return None
    user_msg = (
        "Отредактируй текст внутри тегов <text> согласно правилам из системного "
        "промпта. Выведи ТОЛЬКО итоговый отредактированный текст, без пояснений, "
        "комментариев, приветствий или markdown-форматирования.\n"
        f"<text>\n{text}\n</text>"
    )
    return worker.process(user_msg)


def llm_available() -> bool:
    if LLM_BACKEND == "ollama":
        return _ollama_available()
    if LLM_BACKEND == "api":
        return _api_available()
    return _claude_available()


def llm_process(text: str, system_prompt: str) -> str:
    if not system_prompt:
        return text
    if LLM_BACKEND == "ollama":
        result = ollama_process(text, system_prompt)
    elif LLM_BACKEND == "api":
        if not _api_available():
            log.warning("ANTHROPIC_API_KEY not set — skipping LLM")
            return text
        result = api_process(text, system_prompt)
    else:
        if not _claude_available():
            log.warning("claude CLI not found on PATH — skipping LLM")
            return text
        result = claude_process(text, system_prompt)
    if result is None:
        return text
    return result


# ---------- Recorder ----------
class Recorder:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._stream: Optional[sd.InputStream] = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

    def _cb(self, indata, frames, time_info, status):
        if status:
            log.debug("sd status: %s", status)
        with self._lock:
            self._chunks.append(indata.copy())

    def start(self):
        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._cb,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._chunks, axis=0).flatten()


# ---------- Output ----------
# Scan codes for keys that form our hotkey (`ctrl+shift`).
_CTRL_SCANS  = {29, 97}   # left/right ctrl
_SHIFT_SCANS = {42, 54}   # left/right shift


class HotkeyWatcher:
    """Fires `on_trigger()` only when Ctrl and Shift are pressed together
    cleanly: both held, no other key touched during the hold, then at
    least one of them released. This eliminates races with combos like
    Ctrl+Shift+Left — any foreign key during the hold "dirties" the
    sequence and the trigger is skipped."""

    def __init__(self, on_trigger, min_hold_ms=30, max_hold_ms=2000):
        self.on_trigger = on_trigger
        self.min_hold_sec = min_hold_ms / 1000.0
        self.max_hold_sec = max_hold_ms / 1000.0
        self._ctrl_down = False
        self._shift_down = False
        self._hold_start: Optional[float] = None
        self._dirty = False
        self._lock = threading.Lock()

    def start(self):
        keyboard.hook(self._on_event)
        log.info("hotkey watcher active (ctrl+shift, min %dms, max %dms)",
                 int(self.min_hold_sec * 1000), int(self.max_hold_sec * 1000))

    def _on_event(self, evt):
        down = evt.event_type == "down"
        scan = evt.scan_code
        with self._lock:
            if scan in _CTRL_SCANS:
                self._ctrl_down = down
                self._on_modifier_change()
                return
            if scan in _SHIFT_SCANS:
                self._shift_down = down
                self._on_modifier_change()
                return
            # Any other keydown while we're holding both → combo, not us
            if down and self._hold_start is not None:
                self._dirty = True

    def _on_modifier_change(self):
        both_down = self._ctrl_down and self._shift_down
        if both_down and self._hold_start is None:
            self._hold_start = time.time()
            self._dirty = False
            return
        if not both_down and self._hold_start is not None:
            held = time.time() - self._hold_start
            dirty = self._dirty
            self._hold_start = None
            self._dirty = False
            if dirty:
                log.debug("hotkey hold dirtied — skipping")
                return
            if held < self.min_hold_sec:
                log.debug("hotkey hold too short (%.0fms)", held * 1000)
                return
            if held > self.max_hold_sec:
                log.debug("hotkey hold too long (%.0fms) — ignored", held * 1000)
                return
            # Clean press and release of bare Ctrl+Shift → fire
            threading.Thread(target=self.on_trigger, daemon=True).start()


def _release_mods():
    """Force-release modifier keys to avoid stuck-key states when our
    keyboard ops run while the user still holds the hotkey."""
    for key in ("ctrl", "shift", "alt", "left ctrl", "right ctrl",
                "left shift", "right shift", "left alt", "right alt"):
        try:
            keyboard.release(key)
        except Exception:
            pass


def copy_and_paste(text: str, auto_paste: bool):
    pyperclip.copy(text)
    if auto_paste:
        _release_mods()
        time.sleep(0.05)
        keyboard.send("ctrl+v")


class Spinner:
    """Append-only animated indicator: label is typed once, then a single
    rotating braille char flips in place. Keystrokes per tick: 2 (backspace +
    one char), so desync with the target field is minimized."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    INTERVAL = 0.12
    KEY_DELAY = 0.005

    def __init__(self):
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._label_len = 0
        self._spinner_shown = False
        self._label_lock = threading.Lock()
        self._new_label: Optional[str] = None

    def start(self, label: str):
        self._stop.clear()
        self._done.clear()
        self._label_len = 0
        self._spinner_shown = False
        self._new_label = label
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, label: str):
        with self._label_lock:
            self._new_label = label

    def _write(self, s: str):
        _release_mods()
        keyboard.write(s, delay=self.KEY_DELAY)

    def _backspace(self, n: int):
        if n <= 0:
            return
        _release_mods()
        for _ in range(n):
            keyboard.send("backspace")
            time.sleep(self.KEY_DELAY)

    def _run(self):
        i = 0
        try:
            while not self._stop.is_set():
                with self._label_lock:
                    new_label = self._new_label
                    self._new_label = None

                if new_label is not None:
                    erase = self._label_len + (1 if self._spinner_shown else 0)
                    self._backspace(erase)
                    self._write(new_label)
                    self._label_len = len(new_label)
                    self._spinner_shown = False

                frame = self.FRAMES[i % len(self.FRAMES)]
                if self._spinner_shown:
                    self._backspace(1)
                self._write(frame)
                self._spinner_shown = True
                i += 1

                if self._stop.wait(self.INTERVAL):
                    break
        finally:
            erase = self._label_len + (1 if self._spinner_shown else 0)
            try:
                self._backspace(erase)
            except Exception:
                pass
            self._label_len = 0
            self._spinner_shown = False
            self._done.set()

    def stop(self, timeout: float = 2.0):
        self._stop.set()
        if self._thread is not None:
            self._done.wait(timeout)
            self._thread = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ---------- App ----------
class App:
    def __init__(self):
        self.phase = Phase.IDLE
        self.llm_enabled = LLM_ENABLED
        self.auto_paste = AUTO_PASTE
        self.placeholder = PLACEHOLDER
        self.spinner = Spinner()
        self.asr = ASR(MODEL_NAME)
        self.recorder = Recorder(SAMPLE_RATE)
        self.prompt = load_prompt()
        self.icon: Optional[pystray.Icon] = None
        self.work_q: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # --- tray icon ---
    def _make_icon(self, phase: Phase) -> Image.Image:
        color = {
            Phase.IDLE:       (70, 130, 180),
            Phase.RECORDING:  (220, 50, 50),
            Phase.PROCESSING: (240, 180, 40),
        }[phase]
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((8, 8, 56, 56), fill=color)
        return img

    def _refresh_icon(self):
        if self.icon is not None:
            self.icon.icon = self._make_icon(self.phase)
            self.icon.title = f"fstt — {self.phase.value}"

    def _set_phase(self, p: Phase):
        self.phase = p
        self._refresh_icon()
        log.info("phase -> %s", p.value)

    # --- hotkey handler ---
    def on_hotkey(self):
        if self.phase == Phase.IDLE:
            self._start_recording()
        elif self.phase == Phase.RECORDING:
            self._stop_and_process()
        else:
            log.info("busy (processing), ignoring hotkey")

    def _spinner_enabled(self) -> bool:
        return self.placeholder and self.auto_paste

    def _start_recording(self):
        try:
            self.recorder.start()
        except Exception as e:
            log.error("start recording failed: %s", e)
            return
        self._set_phase(Phase.RECORDING)
        if self._spinner_enabled():
            self.spinner.start("Listening")

    def _stop_and_process(self):
        if self.spinner.active:
            self.spinner.update("Transcribing")
        audio = self.recorder.stop()
        self._set_phase(Phase.PROCESSING)
        self.work_q.put(audio)

    def _worker_loop(self):
        while True:
            audio = self.work_q.get()
            try:
                if audio.size < self.recorder.sample_rate * 0.2:
                    log.info("audio too short, skipping")
                    if self.spinner.active:
                        self.spinner.stop()
                    continue
                text = self.asr.transcribe(audio)
                log.info("ASR: %r", text)
                if not text:
                    if self.spinner.active:
                        self.spinner.stop()
                    continue

                want_llm = self.llm_enabled and self.prompt and llm_available()
                use_stream = (
                    want_llm
                    and LLM_BACKEND == "ollama"
                    and self.auto_paste
                )

                if use_stream:
                    if self.spinner.active:
                        self.spinner.update("Polishing")
                    final = self._stream_and_type(text)
                    if final is None:
                        if self.spinner.active:
                            self.spinner.stop()
                        copy_and_paste(text, self.auto_paste)
                    else:
                        log.info("LLM stream: %r", final)
                        pyperclip.copy(final)
                    continue

                if want_llm:
                    if self.spinner.active:
                        self.spinner.update("Polishing")
                    try:
                        polished = llm_process(text, self.prompt)
                        log.info("LLM: %r", polished)
                        text = polished or text
                    except Exception as e:
                        log.error("LLM failed: %s — using raw", e)
                if self.spinner.active:
                    self.spinner.stop()
                copy_and_paste(text, self.auto_paste)
            except Exception as e:
                log.exception("processing failed: %s", e)
                if self.spinner.active:
                    self.spinner.stop()
            finally:
                self._set_phase(Phase.IDLE)

    def _stream_and_type(self, text: str) -> Optional[str]:
        """Stream Ollama tokens into the active field. Stops the spinner on
        the first chunk, then types each subsequent chunk as it arrives."""
        first = [True]

        def on_chunk(chunk: str):
            if first[0]:
                if self.spinner.active:
                    self.spinner.stop()
                first[0] = False
            _release_mods()
            keyboard.write(chunk, delay=0.003)

        return ollama_stream(text, self.prompt, on_chunk)

    # --- tray menu ---
    def _menu(self):
        def checked_llm(_): return self.llm_enabled
        def checked_paste(_): return self.auto_paste
        def checked_placeholder(_): return self.placeholder

        def toggle_llm(icon, item):
            self.llm_enabled = not self.llm_enabled
            log.info("LLM enabled: %s", self.llm_enabled)

        def toggle_paste(icon, item):
            self.auto_paste = not self.auto_paste
            log.info("auto-paste: %s", self.auto_paste)

        def toggle_placeholder(icon, item):
            self.placeholder = not self.placeholder
            log.info("placeholder: %s", self.placeholder)

        def quit_(icon, item):
            log.info("quitting")
            icon.stop()
            os._exit(0)

        return pystray.Menu(
            pystray.MenuItem("Hotkey: Ctrl+Shift (press & release)", None, enabled=False),
            pystray.MenuItem("LLM polish", toggle_llm, checked=checked_llm),
            pystray.MenuItem("Auto-paste", toggle_paste, checked=checked_paste),
            pystray.MenuItem("Loading dots", toggle_placeholder, checked=checked_placeholder),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", quit_),
        )

    def _warmup_llm(self):
        if not (self.llm_enabled and self.prompt and llm_available()):
            return
        log.info("warming up LLM (%s)...", LLM_BACKEND)
        t0 = time.time()
        try:
            llm_process("тест", self.prompt)
            log.info("LLM warmed in %.1fs", time.time() - t0)
        except Exception as e:
            log.error("warmup failed: %s", e)

    def run(self):
        self._watcher = HotkeyWatcher(
            self.on_hotkey,
            min_hold_ms=HOTKEY_MIN_HOLD_MS,
            max_hold_ms=HOTKEY_MAX_HOLD_MS,
        )
        self._watcher.start()
        threading.Thread(target=self.asr.load, daemon=True).start()
        threading.Thread(target=self._warmup_llm, daemon=True).start()
        self.icon = pystray.Icon(
            "fstt",
            self._make_icon(self.phase),
            "fstt — idle",
            menu=self._menu(),
        )
        self.icon.run()


if __name__ == "__main__":
    try:
        App().run()
    except KeyboardInterrupt:
        sys.exit(0)
