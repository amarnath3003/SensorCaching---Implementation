# llm_backend.py
# llama.cpp subprocess wrapper, measures TTFT

import subprocess, time

class LlamaCppBackend:
    def __init__(self, model_path: str, n_threads: int = 4):
        self.model_path = model_path
        self.n_threads = n_threads

    def generate(self, prompt: str) -> tuple[str, float]:
        """Returns (response_text, ttft_ms)"""
        cmd = [
            "./llama.cpp/build/bin/llama-cli",
            "-m", self.model_path,
            "-p", prompt,
            "-n", "256",
            "--threads", str(self.n_threads),
            "--log-disable"
        ]
        t_start = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        first_token_seen = False
        ttft_ms = None
        response_tokens = []
        for char in iter(lambda: proc.stdout.read(1), ''):
            if not first_token_seen and char.strip():
                ttft_ms = (time.perf_counter() - t_start) * 1000
                first_token_seen = True
            response_tokens.append(char)
        proc.wait()
        return ''.join(response_tokens), ttft_ms or 0.0