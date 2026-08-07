import torch

from causaflux.model import CausaFlux, CausaFluxConfig
from causaflux.synthetic import generate_synthetic_upr


def test_model_forward_probabilistic_outputs_and_loss():
    dataset = generate_synthetic_upr(n_trajectories=8, min_steps=5, max_steps=7, seed=2)
    batch = [dataset[index] for index in range(4)]
    observations = torch.stack([item["observations"] for item in batch])
    observation_mask = torch.stack([item["observation_mask"] for item in batch])
    interventions = torch.stack([item["interventions"] for item in batch])
    times = torch.stack([item["times"] for item in batch])
    mask = torch.stack([item["mask"] for item in batch])
    fates = torch.stack([item["fate"] for item in batch])
    model = CausaFlux(CausaFluxConfig())
    outputs = model(
        observations,
        interventions,
        times,
        mask,
        observation_mask=observation_mask,
        target_observation_mask=observation_mask,
    )
    assert outputs["next_observation_mean"].shape == (4, 6, 12)
    assert outputs["next_observation_log_variance"].shape == (4, 6, 12)
    assert outputs["fate_logits"].shape == (4, 3)
    losses = model.loss(outputs, observations, fates)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
