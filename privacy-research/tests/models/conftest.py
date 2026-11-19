import struct

import pytest


def gguf_string(text: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def gguf_value(value) -> bytes:
    """One metadata value after its type tag: a bool, uint32, float32, string, or list of strings or int32."""
    if isinstance(value, bool):
        return struct.pack("<I?", 7, value)
    if isinstance(value, int):
        return struct.pack("<II", 4, value)
    if isinstance(value, float):
        return struct.pack("<If", 6, value)
    if isinstance(value, str):
        return struct.pack("<I", 8) + gguf_string(value)
    strings = all(isinstance(item, str) for item in value)
    items = b"".join(gguf_string(item) if strings else struct.pack("<i", item) for item in value)
    return struct.pack("<IIQ", 9, 8 if strings else 5, len(value)) + items


@pytest.fixture(scope="session")
def write_gguf():
    """Return a function that writes metadata and float32 tensors as a GGUF version 3 file."""
    def write(path, metadata: dict, tensors: dict):
        header = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata))
        header += b"".join(gguf_string(key) + gguf_value(value) for key, value in metadata.items())
        data = b""
        for name, tensor in tensors.items():
            # GGUF lists dimensions innermost first, the reverse of a numpy shape.
            header += gguf_string(name) + struct.pack(f"<I{tensor.ndim}Q", tensor.ndim, *reversed(tensor.shape))
            header += struct.pack("<IQ", 0, len(data))
            data += tensor.astype("<f4").tobytes()
            data += b"\0" * (-len(data) % 32)
        path.write_bytes(header + b"\0" * (-len(header) % 32) + data)
        return path
    return write
