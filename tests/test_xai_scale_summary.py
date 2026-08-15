from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_bci_sleep_method_scale.py"
SPEC = spec_from_file_location("summarize_bci_sleep_method_scale", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_method_differences_preserve_matching_order_and_seed() -> None:
    rows = {}
    for order in ("forward", "challenging"):
        for seed in (1, 2):
            rows[("sequential", order, seed)] = {
                "relative_forgetting": 0.4,
                "ped": 0.3,
            }
            rows[("ewc", order, seed)] = {
                "relative_forgetting": 0.1,
                "ped": 0.2 if seed == 1 else 0.4,
            }
    result = MODULE.paired_method_differences(rows, "ewc")
    assert result["relative_forgetting_difference"]["mean"] == pytest.approx(-0.3)
    assert result["ped_difference"]["mean"] == pytest.approx(0.0)
    assert result["both_better_count"] == 2
    assert result["performance_better_explanation_worse_count"] == 2
