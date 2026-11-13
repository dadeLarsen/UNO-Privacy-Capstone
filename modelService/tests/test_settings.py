from dataclasses import replace

import pytest
import torch

from model_service.settings import DEFAULT_CONDITIONS, Settings


@pytest.fixture(autouse=True)
def no_service_environment(monkeypatch):
    for name in ("MODEL_SERVICE_CONDITION", "MODEL_SERVICE_DEVICE"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("available, native_bf16, device", [
    (True, True, "cuda"),
    (False, False, "cpu"),
    (True, False, "cpu"),
])
def test_the_default_condition_follows_the_hardware(monkeypatch, available, native_bf16, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    # Like torch, count emulated bf16 as support unless the caller excludes it.
    monkeypatch.setattr(torch.cuda, "is_bf16_supported",
                        lambda including_emulation=True: available and (native_bf16 or including_emulation))
    settings = Settings.from_env()
    assert settings.condition is None
    assert settings.condition_path == DEFAULT_CONDITIONS[device]


def test_a_device_picks_its_default_condition_whatever_the_hardware(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert Settings(device="cuda").condition_path == DEFAULT_CONDITIONS["cuda"]
    monkeypatch.setenv("MODEL_SERVICE_DEVICE", "CPU")
    assert Settings.from_env().condition_path == DEFAULT_CONDITIONS["cpu"]


def test_a_given_condition_is_kept(monkeypatch):
    given = DEFAULT_CONDITIONS["cpu"]
    assert Settings(condition=str(given)).condition_path == given
    monkeypatch.setenv("MODEL_SERVICE_CONDITION", str(given))
    assert Settings.from_env().condition == given


def test_a_condition_plus_a_device_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="a condition or a device, not both"):
        Settings(condition=DEFAULT_CONDITIONS["cpu"], device="cpu")
    monkeypatch.setenv("MODEL_SERVICE_CONDITION", str(DEFAULT_CONDITIONS["cpu"]))
    monkeypatch.setenv("MODEL_SERVICE_DEVICE", "cpu")
    with pytest.raises(ValueError, match="a condition or a device, not both"):
        Settings.from_env()


def test_a_device_flag_alone_is_not_mistaken_for_a_condition_plus_a_device():
    # __main__ applies flags with replace(), which validates again.
    settings = replace(Settings.from_env(), device="cpu")
    assert settings.condition is None
    assert settings.condition_path == DEFAULT_CONDITIONS["cpu"]


def test_a_missing_condition_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        Settings(condition=tmp_path / "missing.toml")
