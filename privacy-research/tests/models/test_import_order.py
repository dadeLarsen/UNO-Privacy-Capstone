"""A backend imports in a fresh interpreter, whichever build of llama.cpp is installed."""

import subprocess
import sys


def test_a_gguf_backend_imports_on_its_own():
    # On Windows the CUDA build of llama.cpp loads only after torch; privacy_bench.models keeps that order.
    result = subprocess.run([sys.executable, "-c", "import privacy_bench.models.gguf_chat_model"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
