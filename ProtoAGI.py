import os
import json
import time
import threading
import queue
import logging
import traceback
from typing import List, Tuple, Optional

MODEL_PATH = r"C:\Users\nitro\Downloads\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MAX_NEW_TOKENS = 2048
MEMORY_MAX_TOKENS = 16000
MEMORY_MAX_ITEMS = 4000
DEFAULT_MODEL_NEW_TOKENS = 512
SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

try:
    from llama_cpp import Llama  # type: ignore
    LLAMA_CPP_AVAILABLE = True
except Exception:
    Llama = None  # type: ignore
    LLAMA_CPP_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False

try:
    import pyttsx3  # type: ignore
    TTS_AVAILABLE = True
except Exception:
    pyttsx3 = None  # type: ignore
    TTS_AVAILABLE = False

try:
    import speech_recognition as sr  # type: ignore
    STT_AVAILABLE = True
except Exception:
    sr = None  # type: ignore
    STT_AVAILABLE = False

try:
    import tkinter as tk  # type: ignore
    from tkinter import END, DISABLED, NORMAL, simpledialog, messagebox  # type: ignore
    TK_AVAILABLE = True
except Exception:
    tk = None  # type: ignore
    END = None
    DISABLED = None
    NORMAL = None
    simpledialog = None
    messagebox = None
    TK_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TinyProto")


class TinyProto:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        memory_file: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
    ):
        self.model_path = model_path
        self.memory_file = memory_file or os.path.join(os.path.dirname(__file__), "tiny_memory.json")
        self.memory: List[str] = self._load_memory()
        self.llama = None
        self.tokenizer = None
        self.model = None
        self._model_lock = threading.RLock()
        try:
            self._load_model()
        except Exception:
            logger.exception("Model load failed during __init__")

    def _load_model(self):
        if LLAMA_CPP_AVAILABLE and self.model_path:
            try:
                n_threads = max(1, (os.cpu_count() or 1) // 2)
                try:
                    self.llama = Llama(model_path=self.model_path, n_ctx=4096, n_threads=n_threads, verbose=False)
                except TypeError:
                    self.llama = Llama(model_path=self.model_path, n_ctx=4096)
                logger.info("Loaded model via llama-cpp-python")
                self.tokenizer = None
                self.model = None
                return
            except Exception as e:
                logger.warning("llama-cpp-python load failed: %s", e)
                self.llama = None

        if TRANSFORMERS_AVAILABLE and self.model_path:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path, low_cpu_mem_usage=True)
                self.llama = None
                logger.info("Loaded model via transformers")
                return
            except Exception as e:
                logger.warning("Transformers load failed: %s", e)
                self.tokenizer = None
                self.model = None
                self.llama = None

        logger.info("No supported model backend available. Running in fallback mode.")
        self.llama = None
        self.tokenizer = None
        self.model = None

    def _load_memory(self) -> List[str]:
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
        except Exception:
            logger.warning("Failed to load memory file, starting fresh.", exc_info=True)
        return []

    def _save_memory(self) -> None:
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("Failed to save memory")

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if LLAMA_CPP_AVAILABLE and getattr(self, "llama", None) is not None:
            try:
                tok = getattr(self.llama, "tokenize", None)
                if callable(tok):
                    parts = tok(text.encode("utf-8"))
                    return len(parts)
            except Exception:
                pass
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return max(1, len(text.split()))

    def _memory_total_tokens(self) -> int:
        total = 0
        for item in self.memory:
            total += self._count_tokens(item)
        return total

    def _prune_memory_if_needed(self) -> None:
        changed = False
        while (len(self.memory) > MEMORY_MAX_ITEMS) or (self._memory_total_tokens() > MEMORY_MAX_TOKENS):
            if not self.memory:
                break
            self.memory.pop(0)
            changed = True
        if changed:
            self._save_memory()

    def remember(self, note: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{ts}] {note}"
        self.memory.append(entry)
        self._prune_memory_if_needed()

    def perceive(self, text: str) -> str:
        return " ".join(text.strip().split())

    def _model_generate(self, prompt: str, max_new_tokens: int = DEFAULT_MODEL_NEW_TOKENS) -> str:
        if LLAMA_CPP_AVAILABLE and getattr(self, "llama", None) is not None:
            try:
                create = getattr(self.llama, "create", None)
                if callable(create):
                    out = self.llama.create(prompt=prompt, max_tokens=max_new_tokens, temperature=0.7)
                    if out and "choices" in out and out["choices"]:
                        text = out["choices"][0].get("text", "") or ""
                        return text.lstrip()
                else:
                    out = self.llama(prompt, max_tokens=max_new_tokens, temperature=0.7)
                    if out and "choices" in out and out["choices"]:
                        text = out["choices"][0].get("text", "") or ""
                        return text.lstrip()
            except Exception:
                logger.debug("llama-cpp generation failed", exc_info=True)

        if self.model is not None and self.tokenizer is not None:
            with self._model_lock:
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
                    )
                    decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
                    if decoded.startswith(prompt):
                        return decoded[len(prompt):].strip()
                    return decoded.strip()
                except Exception:
                    logger.debug("transformers generation failed", exc_info=True)

        return ""

    def _fallback_answer(self, user_text: str) -> str:
        p = user_text.strip()
        if not p:
            return "I don't have enough information to answer. Can you clarify?"
        lower = p.lower()
        if lower.startswith(("what", "who", "when", "where", "how", "why")):
            if "age" in lower:
                return "I don't have an age."
            if "name" in lower:
                return "I'm TinyProto."
            if "you" in lower and "do" in lower:
                return "I can plan, reason, remember, and optionally speak."
            return "I can try to answer that; do you want a brief summary or a detailed explanation?"
        return "Could you be more specific or ask a direct question?"

    def _fallback_reason(self, user_text: str) -> str:
        lower = (user_text or "").lower()
        if "plan" in lower:
            return "I break goals into steps, pick the smallest actionable step, try it, and iterate."
        return "I consider context, recall memory, generate alternatives, weigh them, and select the best."

    def deliberate(self, user_text: str, max_alternatives: int = 3) -> Tuple[List[str], str]:
        prompt_alt = (
            f"Generate {max_alternatives} concise alternative answers (label A/B/C...) to the input below.\n\n"
            f"Input:\n{user_text}\n\n"
            "For each alternative produce one line 'A) <answer>' and a short pros/cons line 'A-pros/cons:'.\n"
            "End with 'Selected: <letter>' and one-line rationale.\n"
            "Keep alternatives concise (<=40 words)."
        )
        deliberation: List[str] = []
        final_answer = ""
        raw = self._model_generate(prompt_alt, max_new_tokens=512)
        if raw:
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            deliberation.extend(lines)
            sel = None
            for l in reversed(lines):
                if l.lower().startswith("selected:"):
                    try:
                        sel = l.split(":", 1)[1].strip()
                        break
                    except Exception:
                        continue
            if sel:
                for l in lines:
                    if l.upper().startswith(sel.upper() + ")") or l.upper().startswith(sel.upper() + "."):
                        final_answer = l.split(")", 1)[-1].strip() if ")" in l else l
                        break
            else:
                for l in lines:
                    if l and (l[0].isalpha() and (len(l) > 1 and l[1] in "). ")):
                        final_answer = l.split(")", 1)[-1].strip() if ")" in l else l
                        break
        if not deliberation:
            t = user_text.strip()
            deliberation = [
                "A) Direct concise answer: " + (self._fallback_answer(t) if t else "No input"),
                "A-pros/cons: fast / may miss nuance",
                "B) Ask a clarifying question: Could you clarify your intent or provide context?",
                "B-pros/cons: avoids wrong assumptions / adds a step",
                "Selected: B",
                "Rationale: Clarifying reduces the risk of incorrect or misleading answers."
            ]
            final_answer = "Could you clarify your request so I can answer accurately?"
        return deliberation, final_answer

    def generate_answer_and_reason(self, user_text: str) -> Tuple[str, str]:
        perceived = user_text.strip()
        deliberation_lines, chosen = self.deliberate(perceived, max_alternatives=3)
        internal_reasoning = "\n".join(deliberation_lines)
        if not chosen:
            chosen = self._fallback_answer(perceived)
        justify_prompt = (
            f"Given the following deliberation:\n{internal_reasoning}\n\n"
            "Provide a 1-2 sentence justification for the selected answer and then the final concise answer.\n"
            "Justification:\nFinal Answer:"
        )
        justification_raw = self._model_generate(justify_prompt, max_new_tokens=256)
        justification = justification_raw.strip() if justification_raw else ""
        if not justification:
            justification = "Justification: Chosen for clarity and likelihood of matching user intent.\nFinal Answer: " + chosen
        if chosen.strip() == perceived or perceived and perceived in chosen:
            chosen = self._fallback_answer(perceived)
        final_answer = chosen.strip()
        final_reasoning = (internal_reasoning + "\n\n" + justification).strip()
        return final_answer, final_reasoning

    def generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        out = self._model_generate(prompt, max_new_tokens=min(max_new_tokens, MAX_NEW_TOKENS))
        if out:
            if out.strip() == prompt.strip() or (prompt.strip() and prompt.strip() in out):
                return ""
            return out
        lower = prompt.lower() if prompt else ""
        if "plan" in lower:
            return "1) Clarify goal. 2) Break down into steps. 3) Execute smallest step. 4) Iterate."
        if any(k in lower for k in ("why", "reason", "think")):
            return "I consider context, recall memory, propose hypotheses, test, and refine."
        return ""

    def plan(self, goal: str) -> List[str]:
        p = f"Plan for: {goal}\nProduce 3 steps."
        raw = self.generate(p)
        steps: List[str] = []
        for line in (raw or "").splitlines():
            s = line.strip()
            if not s:
                continue
            while s and (s[0].isdigit() or s[0] in "-.)"):
                s = s.lstrip("0123456789-.) ")
            if s:
                steps.append(s)
        if not steps and raw:
            parts = [s.strip() for s in raw.split(".") if s.strip()]
            steps = parts[:3] if parts else [goal]
        if not steps:
            steps = [f"Clarify goal for '{goal}'", "Collect required info", "Propose a minimal first action"]
        return steps[:3]

    def act(self, action: str) -> None:
        self.remember(f"Acted: {action}")

    def step(self, user_input: str, remember_reasoning: bool = True) -> Tuple[str, str]:
        perceived = self.perceive(user_input)
        self.remember(f"Perceived: {perceived}")
        answer, reasoning = self.generate_answer_and_reason(perceived)
        if answer and answer != perceived:
            self.remember(f"Answer: {answer}")
        else:
            self.remember("Answer: [no concise answer generated]")
        if remember_reasoning:
            self.remember(f"Reasoning: {reasoning}")
        plan = self.plan(perceived)
        self.remember(f"Plan: {plan}")
        for i, step_text in enumerate(plan, start=1):
            reason = self.generate(f"Step {i}: {step_text}\nContext: {perceived}\nExplain briefly why and how.", max_new_tokens=128)
            self.remember(f"Step {i} Reason: {reason}")
            self.act(f"{i}. {step_text} -> {reason}")
        return answer, reasoning

    def recent_memory(self, n: int = 10) -> List[str]:
        return self.memory[-n:]


def start_tts_engine():
    if not TTS_AVAILABLE:
        return None
    try:
        return pyttsx3.init()
    except Exception:
        logger.exception("TTS init failed")
        return None


def speak_text(engine, text: str):
    if not text:
        return
    if engine:
        def _run():
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                logger.exception("TTS failed during speak")
        threading.Thread(target=_run, daemon=True).start()
    else:
        print("[TTS unavailable] " + text)


def do_stt(recognizer: Optional['sr.Recognizer'], timeout: int = 5, phrase_time_limit: int = 8) -> Tuple[Optional[str], Optional[str]]:
    if not STT_AVAILABLE or recognizer is None:
        return None, "STT library not available"
    try:
        with sr.Microphone() as source:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        try:
            text = recognizer.recognize_google(audio)
            return text, None
        except Exception as e:
            return None, str(e)
    except Exception as e:
        return None, str(e)


def _session_path(name: str) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in "._- ").strip()
    if not safe:
        safe = "session"
    filename = safe + ".json"
    return os.path.join(SESSIONS_DIR, filename)


def list_sessions() -> List[str]:
    items: List[str] = []
    try:
        for f in os.listdir(SESSIONS_DIR):
            if f.lower().endswith(".json"):
                items.append(os.path.splitext(f)[0])
    except Exception:
        logger.exception("list_sessions failed")
    return sorted(items)


def save_session(agent: TinyProto, name: str) -> Tuple[bool, Optional[str]]:
    path = _session_path(name)
    data = {"name": name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "memory": agent.memory}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True, None
    except Exception as e:
        return False, str(e)


def load_session(agent: TinyProto, name: str) -> Tuple[bool, Optional[str]]:
    path = _session_path(name)
    if not os.path.exists(path):
        return False, "session not found"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "memory" in data and isinstance(data["memory"], list):
            agent.memory = data["memory"]
            agent._save_memory()
            return True, None
        return False, "invalid session file"
    except Exception as e:
        return False, str(e)


def delete_session(name: str) -> Tuple[bool, Optional[str]]:
    path = _session_path(name)
    try:
        if os.path.exists(path):
            os.remove(path)
            return True, None
        return False, "not found"
    except Exception as e:
        return False, str(e)


def build_gui(agent: TinyProto):
    if not TK_AVAILABLE:
        print("Tkinter not available. Run in terminal.")
        return

    engine = start_tts_engine()
    recognizer = sr.Recognizer() if STT_AVAILABLE and sr is not None else None

    root = tk.Tk()
    root.title("TinyProto")
    root.geometry("980x720")

    def _tk_error(exc, val, tb):
        logger.error("Tkinter exception: %s", val)
        traceback.print_exception(exc, val, tb)
    root.report_callback_exception = _tk_error

    menubar = tk.Menu(root)
    sessions_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Sessions", menu=sessions_menu)
    root.config(menu=menubar)

    toolbar_frame = tk.Frame(root)
    toolbar_frame.pack(fill="x", padx=6, pady=6)

    tts_var = tk.BooleanVar(value=False)
    tts_check = tk.Checkbutton(toolbar_frame, text="TTS", variable=tts_var)
    tts_check.pack(side="left", padx=4)

    show_thoughts_var = tk.BooleanVar(value=False)
    show_thoughts_check = tk.Checkbutton(toolbar_frame, text="Show Thoughts", variable=show_thoughts_var)
    show_thoughts_check.pack(side="left", padx=4)

    night_var = tk.BooleanVar(value=False)
    night_toggle = tk.Checkbutton(toolbar_frame, text="Night", variable=night_var)
    night_toggle.pack(side="left", padx=4)

    mem_label_var = tk.StringVar(value="Mem tokens: 0 | Items: 0")
    mem_label = tk.Label(toolbar_frame, textvariable=mem_label_var)
    mem_label.pack(side="left", padx=8)

    clear_button = tk.Button(toolbar_frame, text="Clear Memory")
    clear_button.pack(side="right", padx=4)

    log = tk.Text(root, wrap="word", state=DISABLED, height=30)
    log.pack(fill="both", expand=True, padx=6, pady=(0,6))

    entry_frame = tk.Frame(root)
    entry_frame.pack(fill="x", padx=6, pady=(0,6))

    user_var = tk.StringVar()
    entry = tk.Entry(entry_frame, textvariable=user_var)
    entry.pack(side="left", fill="x", expand=True, padx=4, pady=4)
    entry.focus_set()

    mic_button = tk.Button(entry_frame, text="🎤")
    mic_button.pack(side="left", padx=4)

    send_button = tk.Button(entry_frame, text="Send")
    send_button.pack(side="left", padx=4)

    buttons = [tts_check, show_thoughts_check, night_toggle, clear_button, mic_button, send_button]

    light_theme = {
        "root_bg": "#f0f0f0",
        "frame_bg": "#f0f0f0",
        "log_bg": "#ffffff",
        "log_fg": "#000000",
        "entry_bg": "#ffffff",
        "entry_fg": "#000000",
        "button_bg": "#e0e0e0",
        "button_fg": "#000000",
        "menu_bg": "#f0f0f0",
        "menu_fg": "#000000"
    }

    dark_theme = {
        "root_bg": "#141414",
        "frame_bg": "#1f1f1f",
        "log_bg": "#0e0e0e",
        "log_fg": "#e6e6e6",
        "entry_bg": "#2b2b2b",
        "entry_fg": "#ffffff",
        "button_bg": "#2f2f2f",
        "button_fg": "#ffffff",
        "menu_bg": "#1f1f1f",
        "menu_fg": "#e6e6e6"
    }

    def apply_theme(night: bool):
        theme = dark_theme if night else light_theme
        try:
            root.configure(bg=theme["root_bg"])
            toolbar_frame.configure(bg=theme["frame_bg"])
            log.configure(bg=theme["log_bg"], fg=theme["log_fg"], insertbackground=theme["log_fg"])
            entry.configure(bg=theme["entry_bg"], fg=theme["entry_fg"], insertbackground=theme["entry_fg"])
            mem_label.configure(bg=theme["frame_bg"], fg=theme["button_fg"])
            for b in buttons:
                try:
                    b.configure(bg=theme["button_bg"], fg=theme["button_fg"], activebackground=theme["frame_bg"])
                except Exception:
                    pass
        except Exception:
            logger.exception("apply_theme failed")

    night_toggle.configure(command=lambda: apply_theme(night_var.get()))

    def update_mem_label():
        try:
            tokens = agent._memory_total_tokens()
            items = len(agent.memory)
            mem_label_var.set(f"Mem tokens: {tokens} | Items: {items}")
        except Exception:
            mem_label_var.set("Mem tokens: ? | Items: ?")

    def append_log(text: str):
        try:
            log.configure(state=NORMAL)
            log.insert(END, text + "\n")
            log.see(END)
            log.configure(state=DISABLED)
        except Exception:
            logger.exception("append_log failed")

    q = queue.Queue()
    processing_lock = threading.Lock()

    def on_send():
        with processing_lock:
            txt = user_var.get().strip()
            if not txt:
                return
            append_log("User: " + txt)
            entry.delete(0, END)
            send_button.configure(state=DISABLED)
            entry.configure(state=DISABLED)
            threading.Thread(target=run_step_and_update, args=(txt,), daemon=True).start()

    def run_step_and_update(txt: str):
        try:
            answer, reasoning = agent.step(txt, remember_reasoning=True)
        except Exception as e:
            q.put(("error", f"Error during agent.step: {e}"))
            logger.exception("agent.step raised")
            q.put(("action", lambda: (send_button.configure(state=NORMAL), entry.configure(state=NORMAL))))
            return
        q.put(("answer", answer))
        if show_thoughts_var.get():
            q.put(("thoughts", reasoning))
        else:
            q.put(("thoughts-hidden", "Thoughts hidden"))
        q.put(("mem_update", None))
        if tts_var.get():
            speak_text(engine, answer)
        q.put(("action", lambda: (send_button.configure(state=NORMAL), entry.configure(state=NORMAL))))

    def process_queue():
        try:
            while not q.empty():
                kind, payload = q.get_nowait()
                if kind == "answer":
                    append_log("Assistant: " + (payload or "[no answer]"))
                elif kind == "thoughts":
                    append_log("Thoughts: " + (payload or "[no reasoning]"))
                elif kind == "thoughts-hidden":
                    append_log("[Thoughts hidden] (toggle 'Show Thoughts' to reveal)")
                elif kind == "error":
                    append_log("ERROR: " + str(payload))
                elif kind == "mem_update":
                    update_mem_label()
                elif kind == "user":
                    append_log("User (STT): " + (payload or ""))
                    try:
                        user_var.set(payload or "")
                    except Exception:
                        pass
                elif kind == "action":
                    try:
                        if callable(payload):
                            payload()
                    except Exception:
                        logger.exception("queued action failed")
        except Exception:
            logger.exception("process_queue failure")
        finally:
            root.after(100, process_queue)

    def on_mic():
        append_log("Listening...")
        def _stt():
            text, err = do_stt(recognizer)
            if err:
                q.put(("error", "STT error: " + err))
                return
            if text:
                q.put(("user", text))
                q.put(("action", lambda: user_var.set(text)))
        threading.Thread(target=_stt, daemon=True).start()

    def on_forget():
        agent.memory = []
        agent._save_memory()
        update_mem_label()
        append_log("Memory cleared")

    def save_session_prompt():
        name = simpledialog.askstring("Save Session", "Session name:")
        if not name:
            return
        ok, err = save_session(agent, name)
        if ok:
            refresh_sessions_menu()
            append_log(f"Session saved: {name}")
        else:
            messagebox.showerror("Save Error", str(err))

    def load_session_prompt():
        name = simpledialog.askstring("Load Session", "Session name to load:")
        if not name:
            return
        ok, err = load_session(agent, name)
        if ok:
            update_mem_label()
            append_log(f"Session loaded: {name}")
        else:
            messagebox.showerror("Load Error", str(err))

    def delete_session_prompt():
        name = simpledialog.askstring("Delete Session", "Session name to delete:")
        if not name:
            return
        ok, err = delete_session(name)
        if ok:
            refresh_sessions_menu()
            append_log(f"Session deleted: {name}")
        else:
            messagebox.showerror("Delete Error", str(err))

    def quick_load(name: str):
        ok, err = load_session(agent, name)
        if ok:
            update_mem_label()
            append_log(f"Session loaded: {name}")
        else:
            append_log("Load error: " + (err or "unknown"))

    send_button.configure(command=on_send)
    mic_button.configure(command=on_mic)
    clear_button.configure(command=on_forget)
    entry.bind("<Return>", lambda e: on_send())

    def refresh_sessions_menu():
        sessions_menu.delete(0, "end")
        sessions_menu.add_command(label="Save Session...", command=save_session_prompt)
        sessions_menu.add_command(label="Load Session...", command=load_session_prompt)
        sessions_menu.add_command(label="Delete Session...", command=delete_session_prompt)
        sessions_menu.add_separator()
        for name in list_sessions():
            sessions_menu.add_command(label=name, command=lambda n=name: quick_load(n))

    refresh_sessions_menu()
    apply_theme(night_var.get())
    update_mem_label()
    append_log("TinyProto GUI ready")
    root.after(100, process_queue)
    root.mainloop()


if __name__ == "__main__":
    a = TinyProto()
    build_gui(a)


_PAD = """
""" + ("\n".join([f"PAD LINE {i}" for i in range(1, 801)])) + """
"""