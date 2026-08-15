from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_old_importance_controls.py"
SPEC = spec_from_file_location("summarize_old_importance_controls", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_task_identity_accepts_explicit_schema() -> None:
    result = {"old_task": "bciciv2a", "new_task": "sleep_edf_sc"}
    assert MODULE._task_identity_matches(
        result, old_task="bciciv2a", new_task="sleep_edf_sc"
    )


def test_task_identity_accepts_bound_legacy_schema() -> None:
    result = {
        "fisher_sha256": {"bciciv2a": "a", "sleep_edf_sc": "b"},
        "evaluations": {
            "validation": {"bciciv2a": {}, "sleep_edf_sc": {}},
            "test": {"bciciv2a": {}, "sleep_edf_sc": {}},
        },
    }
    assert MODULE._task_identity_matches(
        result, old_task="bciciv2a", new_task="sleep_edf_sc"
    )


def test_task_identity_rejects_unbound_legacy_schema() -> None:
    result = {
        "fisher_sha256": {"bciciv2a": "a", "physionet_mi": "b"},
        "evaluations": {
            "validation": {"bciciv2a": {}, "physionet_mi": {}},
            "test": {"bciciv2a": {}, "physionet_mi": {}},
        },
    }
    assert not MODULE._task_identity_matches(
        result, old_task="bciciv2a", new_task="sleep_edf_sc"
    )
