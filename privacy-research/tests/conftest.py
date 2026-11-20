import pytest
from safetensors.numpy import save_file

from privacy_bench.core.files import digest_bytes, write_json
from privacy_bench.core.fits import FIT_SCHEMA_VERSION, WEIGHTS_FILE, parse_fit_spec


@pytest.fixture
def write_fit():
    """Return a function that writes a complete fit record, in the form the recorded fits have."""
    def write(directory, name, spec_text, report, tensors):
        spec = parse_fit_spec(name, spec_text)
        directory.mkdir(parents=True)
        (directory / "fit.toml").write_text(spec_text, encoding="utf-8")
        save_file(tensors, str(directory / WEIGHTS_FILE))
        write_json(directory / "fit.json", {
            "schema_version": FIT_SCHEMA_VERSION, "task": spec.task, "kind": spec.kind, "fit": spec.as_dict(),
            "fit_sha256": spec.sha256, "status": "complete", "report": report,
            "tensors": {key: list(value.shape) for key, value in sorted(tensors.items())},
            "weights_sha256": digest_bytes((directory / WEIGHTS_FILE).read_bytes()),
        })
        return directory
    return write
