"""
llm_backend.py — llama.cpp subprocess wrapper
Measures TTFT (time-to-first-token) in milliseconds.
Works on both RPi5 (CPU, 4 threads) and Android Termux (CPU, 4 threads).

Paper ref: Section V-A hardware, Table IX, Fig. 7
"""

from __future__ import annotations
import subprocess
import time
import logging
import os
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)

# Prompt template wrapper — improves Gemma-3 instruction following
_PROMPT_TEMPLATE = "<start_of_turn>user\n{query}\n<end_of_turn>\n<start_of_turn>model\n"


class LlamaCppBackend:
    """
    Calls llama-cli as a subprocess.
    Streams stdout character-by-character to capture TTFT precisely.

    Paper: median cache-miss TTFT ~510ms (laptop) / ~2840ms (RPi5)
    """

    def __init__(
        self,
        model_path:  str = config.MODEL_PATH,
        cli_path:    str = config.LLAMA_CLI_PATH,
        n_threads:   int = config.LLAMA_N_THREADS,
        n_tokens:    int = config.LLAMA_MAX_TOKENS,
    ):
        self._model    = model_path
        self._cli      = cli_path
        self._threads  = n_threads
        self._n_tokens = n_tokens
        self._is_ready = self._check_binary()

    def _check_binary(self) -> bool:
        if not Path(self._cli).exists():
            logger.error(
                "llama-cli not found at '%s'. "
                "Build llama.cpp first:\n"
                "  git clone https://github.com/ggerganov/llama.cpp\n"
                "  cd llama.cpp && cmake -B build && cmake --build build -j4",
                self._cli
            )
            return False
        if not Path(self._model).exists():
            logger.error(
                "Model file not found: '%s'\n"
                "Download Gemma-3-1B-IT-Q4_K_M.gguf from HuggingFace:\n"
                "  https://huggingface.co/bartowski/gemma-3-1b-it-GGUF",
                self._model
            )
            return False
        logger.info("LlamaCppBackend ready: model=%s threads=%d",
                    Path(self._model).name, self._threads)
        return True

    def generate(self, query: str) -> tuple[str, float]:
        """
        Run inference. Returns (response_text, ttft_ms).
        Raises RuntimeError if binary/model not available.
        """
        if not self._is_ready:
            raise RuntimeError(
                "llama-cli or model not found. "
                f"model={self._model} cli={self._cli}"
            )

        prompt = _PROMPT_TEMPLATE.format(query=query)

        cmd = [
            self._cli,
            "--model",        self._model,
            "--prompt",       prompt,
            "--n-predict",    str(self._n_tokens),
            "--threads",      str(self._threads),
            "--log-disable",
            "--no-display-prompt",
            "--temp",         "0.7",
            "--top-p",        "0.9",
        ]

        t_start          = time.perf_counter()
        first_token_seen = False
        ttft_ms: float   = 0.0
        tokens: list[str] = []

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
            )

            # Stream char-by-char to capture TTFT
            for char in iter(lambda: proc.stdout.read(1), ""):
                if not first_token_seen and char.strip():
                    ttft_ms          = (time.perf_counter() - t_start) * 1000
                    first_token_seen = True
                tokens.append(char)

            proc.wait()

            if proc.returncode != 0:
                err = proc.stderr.read()
                logger.warning("llama-cli exit=%d stderr=%s", proc.returncode, err[:200])

        except Exception as exc:
            raise RuntimeError(f"llama.cpp inference failed: {exc}") from exc

        response = "".join(tokens).strip()
        logger.debug("LLM: TTFT=%.1fms len=%d chars", ttft_ms, len(response))
        return response, ttft_ms or (time.perf_counter() - t_start) * 1000


class MockLLMBackend:
    """
    Fast mock backend for unit-testing the eval harness
    without running the full LLM. Returns deterministic responses.
    """

    def generate(self, query: str) -> tuple[str, float]:
        import random, time
        time.sleep(0.001)   # simulate ~1ms cache-hit latency reference point
        # Simulate inference latency
        ttft = random.gauss(510, 30)   # ~510ms as per Table IX
        return f"[MOCK RESPONSE to: {query[:40]}]", max(400.0, ttft)
