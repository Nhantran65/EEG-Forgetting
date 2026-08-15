from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_task_attribution_gate.py"
SPEC = spec_from_file_location("summarize_task_attribution_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_task_gate_requires_two_of_three_per_hard_gate_and_all_positive_ig() -> None:
    rows = [
        {"fidelity_passed": True, "drift_above_null_passed": True, "integrated_gradients_all_positive": True},
        {"fidelity_passed": True, "drift_above_null_passed": False, "integrated_gradients_all_positive": True},
        {"fidelity_passed": False, "drift_above_null_passed": True, "integrated_gradients_all_positive": True},
    ]
    assert MODULE.task_gate_decision(rows)["scale_allowed"] is True
    rows[-1]["integrated_gradients_all_positive"] = False
    assert MODULE.task_gate_decision(rows)["scale_allowed"] is False
