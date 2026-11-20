"""Model backends: load a pinned local GGUF file, run one input, and measure it.

Backends know nothing about tasks. They take text or token IDs and return model outputs
(generated text, embeddings); strategies inside a task turn those outputs into task answers.
"""

import ctypes
from importlib import metadata
import sys

# What the CUDA build of llama.cpp needs and does not ship: each NVIDIA package of the cu132 group, with its
# libraries in the order they load. cublas imports cublasLt. The Windows build holds the CUDA runtime itself.
CUDA_LIBRARIES = {
    "win32": {"nvidia-cublas": ("cublasLt64_13.dll", "cublas64_13.dll")},
    "linux": {"nvidia-cuda-runtime": ("libcudart.so.13",),
              "nvidia-cublas": ("libcublasLt.so.13", "libcublas.so.13")},
}


def load_cuda_libraries():
    """Load NVIDIA's libraries by their full path, when their packages are installed.

    The packages put them in a folder that the system's loader does not search, so `import llama_cpp` fails
    on the CUDA build unless the libraries are in the process already.
    """
    for package, names in CUDA_LIBRARIES.get(sys.platform, {}).items():
        try:
            files = metadata.files(package)
        except metadata.PackageNotFoundError:
            # The cpu group installs a build of llama.cpp without CUDA, and none of these packages.
            return
        paths = {file.name: file.locate() for file in files}
        for name in names:
            ctypes.CDLL(str(paths[name]))


# Every backend imports llama_cpp, and Python runs this file before any of them.
load_cuda_libraries()
