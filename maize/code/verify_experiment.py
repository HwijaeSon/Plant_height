"""Check saved models, test-target independence, and process integration."""
import json
from pathlib import Path

import numpy as np
import torch

from data import make_dataset, ROOT
from run_experiment import build_model, forward, tensor_batch, temperature_response, logistic


def main():
    root = ROOT / "results/chronological_final_seed1_3"
    torch.set_num_threads(2)
    ds = make_dataset()
    device = torch.device("cpu")
    batch = tensor_batch(ds.test.subset(np.array([0, 200, 401])), device)
    tau = torch.linspace(0, 1, len(ds.days))
    params = json.loads((root / "logistic_parameters.json").read_text())
    restored = []
    for key in ["latent", "lstm", "pinn"]:
        for seed in [1, 2, 3]:
            folder = root / "runs" / f"{key}_seed{seed}"
            saved = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
            model = build_model(key, ds, params if key == "pinn" else None)
            model.load_state_dict(saved["state_dict"])
            model.eval()
            def predict(b):
                out = forward(key, model, b, tau)
                return out["pred"] if key == "latent" else out["prediction"]
            with torch.no_grad():
                value = predict(batch)
                altered = dict(batch, y=batch["y"]*10+99, mask=1-batch["mask"])
                other = predict(altered)
            np.testing.assert_array_equal(value.numpy(), other.numpy())
            pred = np.load(folder / "predictions.npz", allow_pickle=False)["test"][[0, 200, 401]]
            np.testing.assert_allclose(value.numpy(), pred, atol=2e-5, rtol=2e-5)
            if key == "pinn":
                assert saved["best_epoch"] > 500
            restored.append(f"{key}_seed{seed}")
    # Constant temperature reduces the daily integral to response * elapsed days.
    response = temperature_response(np.full(96, 20., dtype=np.float32), 292., 303.)
    integral = np.r_[0., np.cumsum((response[:-1]+response[1:])/2)]
    np.testing.assert_allclose(integral, response[0]*np.arange(96), atol=1e-12)
    delta = temperature_response(np.array([20.], dtype=np.float32), 292.+1e-5, 303.) - temperature_response(np.array([20.], dtype=np.float32), 292., 303.)
    assert delta[0] != 0 and delta.dtype == np.float64
    np.testing.assert_allclose(logistic(0., .15, .8), 1e-4)
    # A deterministic process fit has one coefficient row per genotype, from train only.
    assert params["genotypes"] == ds.genotypes
    assert all(f["success"] for f in params["fits"])
    checks = dict(status="passed", restored_neural_checkpoints=restored,
                  checkpoint_predictions="match saved GPU outputs on 3 test curves within CPU/GPU tolerance",
                  test_target_independence="changing test heights and masks leaves all 9 models' predictions identical",
                  pinn_checkpoints="all after physics warm-up", temperature_integration="constant-temperature identity passed",
                  optimizer_precision="float64 temperature parameter perturbations are nonzero",
                  process_initial_condition="fixed H0 identity passed")
    (root / "model_validation.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
