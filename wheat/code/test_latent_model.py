import torch

from model import LatentODEHeightModel, masked_rmse_loss


def test_forward_shape_and_gradients() -> None:
    torch.manual_seed(0)
    model = LatentODEHeightModel(
        n_genotypes=5,
        env_dim=1,
        latent_dim=4,
        g_embed_dim=3,
        ode_hidden=6,
        ode_layers=1,
        dec_hidden=5,
        enc_hidden=4,
        genetic_encoding="one_hot",
        days_per_tau=7.0,
        use_physics=False,
    )
    genotype = torch.tensor([0, 3])
    environment = torch.randn(2, 8, 1)
    time = torch.linspace(0, 1, 8).expand(2, -1)
    output = model(genotype, environment, time, time[:, 1:] - time[:, :-1], 0)

    assert output["z"].shape == (2, 8, 4)
    assert output["pred"].shape == (2, 8)
    target = torch.rand(2, 8)
    mask = torch.ones_like(target)
    masked_rmse_loss(output["pred"], target, mask).backward()
    assert next(model.ode_func.parameters()).grad is not None


def test_kinship_encoding() -> None:
    model = LatentODEHeightModel(
        n_genotypes=3,
        env_dim=1,
        latent_dim=4,
        g_embed_dim=2,
        genetic_encoding="kinship",
        kinship=torch.eye(3),
        use_physics=False,
    )
    environment = torch.randn(1, 5, 1)
    time = torch.linspace(0, 1, 5).unsqueeze(0)
    output = model(torch.tensor([1]), environment, time, time[:, 1:] - time[:, :-1], 0)
    assert output["pred"].shape == (1, 5)
