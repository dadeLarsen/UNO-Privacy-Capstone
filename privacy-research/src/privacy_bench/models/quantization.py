"""bitsandbytes settings applied at load time, shared by every quantizing backend."""

import logging

from transformers import BitsAndBytesConfig

INT8_CAST_LOGGER = "bitsandbytes.autograd._functions"
INT8_CAST_MESSAGE = "MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"


def bitsandbytes_config(precision: str, compute_dtype) -> BitsAndBytesConfig:
    """INT8 with outlier threshold 6.0, or NF4 with 64-value blocks and no double quantization."""
    if precision not in ("int8", "nf4"):
        raise ValueError(f"Quantized precision must be int8 or nf4, not {precision!r}")
    return BitsAndBytesConfig(
        load_in_8bit=precision == "int8", load_in_4bit=precision == "nf4",
        llm_int8_threshold=6.0, llm_int8_enable_fp32_cpu_offload=False,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )


class _OnceInt8CastMessage(logging.Filter):
    """Show one expected bitsandbytes notice once per process; keep every other message."""

    def __init__(self):
        super().__init__()
        self.seen = False

    def filter(self, record):
        if (record.name != INT8_CAST_LOGGER or record.levelno != logging.WARNING
                or record.getMessage() != INT8_CAST_MESSAGE):
            return True
        if self.seen:
            return False
        self.seen = True
        return True


def install_int8_cast_log_filter():
    # bitsandbytes 0.50.2 reports the BF16-to-FP16 cast through logger.warning on every forward pass.
    logger = logging.getLogger(INT8_CAST_LOGGER)
    if not any(isinstance(item, _OnceInt8CastMessage) for item in logger.filters):
        logger.addFilter(_OnceInt8CastMessage())
