from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class CausaFluxConfig:
    observation_dim: int = 12
    intervention_dim: int = 4
    hidden_dim: int = 96
    identity_dim: int = 12
    adaptation_dim: int = 24
    commitment_dim: int = 12
    context_dim: int = 12
    time_dim: int = 12
    dropout: float = 0.15
    n_fates: int = 3
    time_scale: float = 1.0
    min_log_variance: float = -6.0
    max_log_variance: float = 2.0

    def to_dict(self) -> dict:
        return asdict(self)


class TimeEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.SiLU(),
        )

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        dt = dt.clamp_min(0.0).unsqueeze(-1)
        features = torch.cat(
            [
                dt,
                torch.log1p(dt),
                torch.sin(dt),
                torch.cos(dt),
                torch.sin(0.5 * dt),
                torch.cos(0.5 * dt),
                torch.sqrt(dt + 1e-6),
            ],
            dim=-1,
        )
        return self.net(features)


class CausaFlux(nn.Module):
    """Factorized probabilistic model for irregular stress-adaptation trajectories."""

    def __init__(self, config: CausaFluxConfig | None = None) -> None:
        super().__init__()
        self.config = config or CausaFluxConfig()
        c = self.config
        observed_input_dim = c.observation_dim * 2
        self.time_encoder = TimeEncoder(c.time_dim)
        self.identity_encoder = nn.Sequential(
            nn.Linear(observed_input_dim, c.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(c.hidden_dim // 2, c.identity_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(observed_input_dim, c.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(c.hidden_dim // 2, c.context_dim),
        )
        self.initial_hidden = nn.Linear(c.identity_dim + c.context_dim, c.hidden_dim)
        self.gru = nn.GRUCell(
            observed_input_dim + c.intervention_dim + c.time_dim,
            c.hidden_dim,
        )
        self.adaptation_head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.adaptation_dim),
            nn.Tanh(),
        )
        self.commitment_increment = nn.Sequential(
            nn.Linear(c.hidden_dim, c.commitment_dim),
            nn.Softplus(),
        )
        latent_dim = c.identity_dim + c.adaptation_dim + c.commitment_dim + c.context_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + c.intervention_dim + c.time_dim, c.hidden_dim),
            nn.SiLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.SiLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, c.observation_dim * 2),
        )
        self.fate_head = nn.Sequential(
            nn.Linear(latent_dim, c.hidden_dim),
            nn.SiLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, c.n_fates),
        )

    def _masked_observation(
        self, observation: torch.Tensor, observation_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([observation * observation_mask, observation_mask], dim=-1)

    def _initial_state(
        self,
        first_observation: torch.Tensor,
        first_observation_mask: torch.Tensor | None = None,
    ):
        if first_observation_mask is None:
            first_observation_mask = torch.ones_like(first_observation)
        encoded_input = self._masked_observation(first_observation, first_observation_mask)
        identity = self.identity_encoder(encoded_input)
        context = self.context_encoder(encoded_input)
        hidden = torch.tanh(self.initial_hidden(torch.cat([identity, context], dim=-1)))
        commitment = first_observation.new_zeros(
            first_observation.size(0), self.config.commitment_dim
        )
        return identity, context, hidden, commitment

    def step(
        self,
        observation: torch.Tensor,
        observation_mask: torch.Tensor,
        intervention: torch.Tensor,
        dt: torch.Tensor,
        identity: torch.Tensor,
        context: torch.Tensor,
        hidden: torch.Tensor,
        commitment: torch.Tensor,
    ):
        time_embedding = self.time_encoder(dt / self.config.time_scale)
        hidden = self.gru(
            torch.cat(
                [
                    self._masked_observation(observation, observation_mask),
                    intervention,
                    time_embedding,
                ],
                dim=-1,
            ),
            hidden,
        )
        adaptation = self.adaptation_head(hidden)
        increment = self.commitment_increment(hidden) * dt.clamp_min(0).unsqueeze(-1)
        commitment = commitment + 0.05 * increment
        latent = torch.cat([identity, adaptation, commitment, context], dim=-1)
        decoded = self.decoder(torch.cat([latent, intervention, time_embedding], dim=-1))
        next_mean, next_log_variance = torch.chunk(decoded, 2, dim=-1)
        next_log_variance = next_log_variance.clamp(
            self.config.min_log_variance, self.config.max_log_variance
        )
        return (
            next_mean,
            next_log_variance,
            latent,
            hidden,
            commitment,
            adaptation,
        )

    def forward(
        self,
        observations: torch.Tensor,
        interventions: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
        target_observation_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if observation_mask is None:
            observation_mask = torch.ones_like(observations)
        if target_observation_mask is None:
            target_observation_mask = observation_mask
        batch, steps, _ = observations.shape
        identity, context, hidden, commitment = self._initial_state(
            observations[:, 0], observation_mask[:, 0]
        )
        means = []
        log_variances = []
        adaptations = []
        commitments = []
        latents = []

        for t in range(steps - 1):
            dt = (times[:, t + 1] - times[:, t]).clamp_min(0.0)
            (
                mean,
                log_variance,
                latent,
                hidden_new,
                commitment_new,
                adaptation,
            ) = self.step(
                observations[:, t],
                observation_mask[:, t],
                interventions[:, t],
                dt,
                identity,
                context,
                hidden,
                commitment,
            )
            active = mask[:, t].unsqueeze(-1)
            hidden = hidden_new * active + hidden * (1.0 - active)
            commitment = commitment_new * active + commitment * (1.0 - active)
            means.append(mean)
            log_variances.append(log_variance)
            adaptations.append(adaptation)
            commitments.append(commitment)
            latents.append(latent)

        mean_tensor = torch.stack(means, dim=1)
        log_variance_tensor = torch.stack(log_variances, dim=1)
        adaptation_tensor = torch.stack(adaptations, dim=1)
        commitment_tensor = torch.stack(commitments, dim=1)
        latent_tensor = torch.stack(latents, dim=1)

        transition_mask = mask[:, 1:] * mask[:, :-1]
        target_mask = target_observation_mask[:, 1:] * transition_mask.unsqueeze(-1)
        last_index = transition_mask.sum(dim=1).long().clamp_min(1) - 1
        batch_indices = torch.arange(batch, device=observations.device)
        final_latent = latent_tensor[batch_indices, last_index]
        fate_logits = self.fate_head(final_latent)

        return {
            "next_observation_mean": mean_tensor,
            "next_observation_log_variance": log_variance_tensor,
            "next_observation": mean_tensor,
            "fate_logits": fate_logits,
            "identity": identity,
            "context": context,
            "adaptation": adaptation_tensor,
            "commitment": commitment_tensor,
            "latent": latent_tensor,
            "transition_mask": transition_mask,
            "target_observation_mask": target_mask,
        }

    def loss(
        self,
        outputs: dict[str, torch.Tensor],
        observations: torch.Tensor,
        fates: torch.Tensor,
        state_weight: float = 1.0,
        fate_weight: float = 0.5,
        smoothness_weight: float = 0.01,
        variance_weight: float = 0.01,
        fate_class_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        target = observations[:, 1:]
        mask = outputs["target_observation_mask"]
        error = outputs["next_observation_mean"] - target
        inverse_variance = torch.exp(-outputs["next_observation_log_variance"])
        gaussian_nll = 0.5 * (
            inverse_variance * error.pow(2) + outputs["next_observation_log_variance"]
        )
        state_nll = (gaussian_nll * mask).sum() / (mask.sum() + 1e-8)
        state_mse = (error.pow(2) * mask).sum() / (mask.sum() + 1e-8)
        fate_loss = F.cross_entropy(
            outputs["fate_logits"], fates, weight=fate_class_weights
        )
        if outputs["adaptation"].size(1) > 1:
            delta = outputs["adaptation"][:, 1:] - outputs["adaptation"][:, :-1]
            smooth_mask = outputs["transition_mask"][:, 1:].unsqueeze(-1)
            smoothness = (delta.pow(2) * smooth_mask).sum() / (
                smooth_mask.sum() * delta.size(-1) + 1e-8
            )
        else:
            smoothness = state_nll.new_tensor(0.0)
        variance_penalty = outputs["next_observation_log_variance"].pow(2).mean()
        total = (
            state_weight * state_nll
            + fate_weight * fate_loss
            + smoothness_weight * smoothness
            + variance_weight * variance_penalty
        )
        return {
            "loss": total,
            "state_nll": state_nll,
            "state_mse": state_mse,
            "fate_loss": fate_loss,
            "smoothness_loss": smoothness,
            "variance_penalty": variance_penalty,
        }
