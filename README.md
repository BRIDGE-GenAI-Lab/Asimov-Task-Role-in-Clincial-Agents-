# Advocacy instructions, assigned scope and adherence to shared-resource allocation rules by clinical AI agents

Code, scenarios, results and figures for the study *"Advocacy instructions, assigned scope and adherence to
shared-resource allocation rules by clinical AI agents: a controlled, agent-based experimental study of 24 large
language models"* (BRIDGE GenAI Lab, Beth Israel Deaconess Medical Center, Harvard Medical School).

A simulated hospital ward gives a tested language-model agent and scripted peers three shared resources (one
ventilator, an imaging queue and an on-call attending). Only the agent's role instruction changes across conditions;
the patients, resources and action tools stay fixed. Every action is scored by a fixed program against the ward's
allocation policy, with no model judge.

- **Experiment 1** (20 models from 15 developers): five role instructions, with no allocation policy stated.
- **Experiment 2** (21 models, four of them high-capability): every role was shown the policy on matched scenario
  pairs in which the policy favoured either the agent's own patient or a competing patient. It also tested whether
  agents could name the patient the policy favours, asked for a judgment before the action, placed a live allocation
  check between agent and resource, repeated the sampling, repeated experiment 1 on the high-capability models and,
  in five models, hid the policy.

## Main results

With the policy shown, assigning one patient rather than the whole ward increased inappropriate ventilator claims by
5.0 percentage points (95% CI 1.1 to 8.9; primary outcome). Inappropriate claims were 11.9% for patient
responsibility under the ward's allocation priority and 45.9% for strong advocacy (33.9 points, 95% CI 17.7 to
50.1); appropriate claims were 94.8% and 90.7%. A sentence giving the policy precedence changed inappropriate claims
by -27.9 points (95% CI -44.0 to -11.8). Without a stated policy (experiment 1), acquisitions discordant with the
study's standard rose from 32.5% to 69.5% between the same roles (36.9 points, 95% CI 25.8 to 48.1).

## Layout

```
code/scarcity_ward/      simulated ward, scorer, model adapters, prompts and harm coefficients
code/analysis/           shared statistics helpers
data/scenarios/          synthetic scenario sets (no patient records)
experiment1/results/     results of experiment 1 (JSON)
experiment2/results/     results of experiment 2 (JSON)
experiment2/runs/        matched-pair scenario set and scripted reference policies of experiment 2
scripts/                 experiment runners, scenario generator and figure generators
scripts/analysis/        analyses that write the results files
figures/                 figures 1-5 and appendix figures 1-4
tests/                   test suite
DATA_DICTIONARY.md       fields of the episode logs
```

## Install and test

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Hosted models were queried through OpenRouter, each pinned to one provider; in experiment 1, open-weight models were
also served locally (Ollama or llama.cpp). Set the API keys for any hosted model in a `.env` file. Analyses were run
on Python 3.9.6 with NumPy 2.0.2, SciPy 1.13.1, statsmodels 0.14.6 and pandas 2.3.3.

## Results files

| File | Written by |
|---|---|
| `experiment2/results/experiment2_primary.json` | `scripts/analysis/experiment2_primary.py` (main estimates) |
| `experiment2/results/experiment2_secondary.json` | `scripts/analysis/experiment2_secondary.py` (sensitivity analyses) |
| `experiment2/results/experiment2_high_capability.json` | `scripts/analysis/experiment2_high_capability.py` (high-capability models) |
| `experiment2/results/experiment2_clusters.json` | `scripts/analysis/experiment2_clusters.py` (developer-clustered intervals) |
| `experiment2/results/experiment2_replicates.json` | `scripts/analysis/experiment2_replicates.py` (repeated sampling) |
| `experiment2/results/experiment2_policy_hidden.json` | `scripts/analysis/experiment2_policy_hidden.py` (policy hidden; experiment 1 design on the high-capability models) |
| `experiment2/results/experiment1_reanalysis.json` | `scripts/analysis/experiment1_reanalysis.py` (experiment 1 role rates and margins on the experiment 2 scale) |
| `experiment2/results/ed_reference.json` | `scripts/analysis/ed_reference.py` (emergency-department scenario family) |
| `experiment1/results/*.json` | the scripts of the same name in `scripts/analysis/` |

## Figures

| Figure | Generator |
|---|---|
| Figure 1 | `scripts/fig1_scope_of_concern.py` |
| Figures 2-5, appendix figure 2 | `scripts/make_figures.py` |
| Appendix figure 1 | `scripts/fig4_words_ride_prior.py` |
| Appendix figure 3 | `scripts/fig_scope2x2_forest.py` |
| Appendix figure 4 | `scripts/fig_model_arm_matrix.py` |

`python3 scripts/make_figures.py` draws all nine figures from the results files; the figure style is in
`scripts/_figstyle.py`.

## Data

Scenarios were derived from MIMIC-IV and MIMIC-IV-ED, credentialed-access PhysioNet datasets that may not be
redistributed. The scenario sets in `data/scenarios/` and `experiment2/runs/` are synthetic and contain no patient
records. The per-episode logs are not stored in this repository; their fields are described in `DATA_DICTIONARY.md`.

## License

MIT (see `LICENSE`).
