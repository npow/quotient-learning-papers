"""End-to-end certified pipeline: held-out independent blocks, finite-sample
TV test, power gate, and a validated sidecar next to the result."""

import json

from experiments.main_table.run_certified import run_one
from quotient_learning.config import CertifiedConfig


class TestCertifiedPipeline:
    def test_tiger_run_writes_certified_sidecar(self, tmp_path):
        row = run_one("tiger", seed=0, out_dir=tmp_path)
        assert row["env"] == "tiger"
        assert row["n_classes"] >= 1
        sidecar_path = tmp_path / "tiger_seed0.certified.json"
        data = json.loads(sidecar_path.read_text())
        assert data["certified"] is True
        assert data["test"] == "TV_FIXED_ALPHABET"
        assert data["sampling_scheme"] == "INDEPENDENT_TRAJECTORY_BLOCK"
        assert data["selection_fraction"] == 0.5
        assert data["stride"] == 0
        assert data["seeds"] == [0]
        assert data["python"] and data["packages"].get("numpy")
        # Round-trips through the validating loader.
        cfg = CertifiedConfig.load_sidecar(sidecar_path)
        assert cfg.stride == data["stride"]

    def test_loader_refuses_uncertified_sidecar(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"certified": False, "k": 2, "tau": 1}))
        try:
            CertifiedConfig.load_sidecar(bad)
        except ValueError as e:
            assert "not certified" in str(e)
        else:
            raise AssertionError("loader accepted an uncertified sidecar")
