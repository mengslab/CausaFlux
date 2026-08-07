# CausaFlux v1.9.0 — AI Model Router

The v1.9 model router integrates validated CausaFlux modules rather than selecting a single model solely by in-sample accuracy.

## Components

| Component | Role | Validation source |
|---|---|---|
| Dynamic state | irregular-time future-state and fate forecasting | held-out perturbation histories |
| Multimodal fusion | early imaging/reporter + destructive multimodal inference | held-out multimodal trajectories |
| Intervention generalization | unseen perturbation, dose, combination and sequence response | intervention holdouts |
| Spatiotemporal context | neighborhood-conditioned future state | held-out sections and donors |
| Foundation representation | transferable pretrained representation | donor/tissue/perturbation holdouts |
| Prospective calibration | interval and outcome calibration | locked Cycle 1→2→3 reference |

Each module receives a bounded reliability score and normalized ensemble weight. Module disagreement is retained as epistemic uncertainty rather than averaged away.

## Intervention objective

The reference ranking combines recovery potential, proteostasis capacity, mitochondrial reserve, inflammatory dysfunction, commitment risk, predictive uncertainty and experiment/intervention cost. The ranking is therefore a decision-support output rather than a raw state forecast.

## Model transparency

`ai/ai_model_router.csv` records the selected model, primary metric, reliability, normalized weight, evidence class and prospective relevance for every component.
