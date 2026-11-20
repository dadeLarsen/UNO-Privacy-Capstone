"""A backend imports in a fresh interpreter, whichever build of llama.cpp is installed."""

import subprocess
import sys


def test_a_gguf_backend_imports_on_its_own():
    # The CUDA build of llama.cpp loads only after NVIDIA's libraries; privacy_bench.models loads them first.
    result = subprocess.run([sys.executable, "-c", "import privacy_bench.models.gguf_chat_model"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
