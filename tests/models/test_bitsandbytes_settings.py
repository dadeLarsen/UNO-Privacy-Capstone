import logging

import pytest
import torch

from privacy_bench.models import quantization


@pytest.fixture
def cast_logger():
    logger = logging.getLogger(quantization.INT8_CAST_LOGGER)
    saved = (logger.handlers[:], logger.filters[:], logger.level, logger.propagate, logger.disabled)
    yield logger
    logger.handlers, logger.filters = saved[0], saved[1]
    logger.setLevel(saved[2])
    logger.propagate, logger.disabled = saved[3], saved[4]


def test_int8_cast_message_is_logged_once_and_other_messages_survive(cast_logger):
    emitted = []
    handler = logging.StreamHandler()
    handler.emit = lambda record: emitted.append(record.getMessage())
    cast_logger.handlers, cast_logger.filters = [handler], [quantization._OnceInt8CastMessage()]
    cast_logger.setLevel(logging.WARNING)
    cast_logger.propagate, cast_logger.disabled = False, False
    for _ in range(3):
        cast_logger.warning("MatMul8bitLt: inputs will be cast from %s to float16 during quantization", "torch.bfloat16")
    cast_logger.warning("Unexpected quantization warning")
    cast_logger.warning("MatMul8bitLt: inputs will be cast from %s to float16 during quantization", "torch.float32")
    cast_logger.error(quantization.INT8_CAST_MESSAGE)
    assert emitted == [quantization.INT8_CAST_MESSAGE, "Unexpected quantization warning",
                       "MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization",
                       quantization.INT8_CAST_MESSAGE]


def test_cast_filter_install_is_idempotent_and_keeps_its_seen_state(cast_logger):
    cast_logger.filters = []
    quantization.install_int8_cast_log_filter()
    limiter = cast_logger.filters[0]
    limiter.seen = True
    quantization.install_int8_cast_log_filter()
    assert cast_logger.filters == [limiter] and limiter.seen


def test_bitsandbytes_settings_match_the_recorded_runs():
    int8 = quantization.bitsandbytes_config("int8", torch.bfloat16)
    assert (int8.load_in_8bit, int8.load_in_4bit, int8.llm_int8_threshold) == (True, False, 6.0)
    assert not int8.llm_int8_enable_fp32_cpu_offload and not int8.llm_int8_has_fp16_weight
    nf4 = quantization.bitsandbytes_config("nf4", torch.bfloat16)
    assert (nf4.load_in_4bit, nf4.bnb_4bit_quant_type, nf4.bnb_4bit_compute_dtype) == (True, "nf4", torch.bfloat16)
    assert not nf4.bnb_4bit_use_double_quant and nf4.bnb_4bit_quant_storage == torch.uint8
    with pytest.raises(ValueError, match="int8 or nf4"):
        quantization.bitsandbytes_config("fp4", torch.bfloat16)
