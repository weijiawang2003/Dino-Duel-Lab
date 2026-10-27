"""Statistical regression: reused environment seeds are shared test conditions."""
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_trained import hierarchical_summary


def test_duplicate_training_rows_do_not_shrink_shared_environment_uncertainty():
    # All policies have exactly the same successes and failures on the same
    # 50 environments. Repeating that row adds no independent environmental
    # evidence, so its confidence interval must not shrink as if it did.
    row = np.r_[np.zeros(25), np.ones(25)]
    two = hierarchical_summary(np.tile(row, (2, 1)), [11, 23], samples=5000, seed=812)
    twelve = hierarchical_summary(np.tile(row, (12, 1)), list(range(12)), samples=5000, seed=812)
    a, b = np.asarray(two["crossed_ci95"]), np.asarray(twelve["crossed_ci95"])
    assert two["hierarchical_ci95"] == two["crossed_ci95"]
    assert twelve["hierarchical_ci95"] == twelve["crossed_ci95"]
    assert two["mean"] == twelve["mean"] == 0.5
    assert two["bootstrap_design"] == twelve["bootstrap_design"] == "crossed_training_and_shared_environment_seeds"
    # Small tolerance admits Monte Carlo quantile variation. A nested sampler
    # incorrectly treating rows' shared environments as independent fails this.
    np.testing.assert_allclose(a, b, atol=0.02, rtol=0)
    assert b[1] - b[0] >= 0.24
