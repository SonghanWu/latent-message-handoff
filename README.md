# What Should a Latent Message Preserve?

Submission for the DAPLab **Latent-comm** task.

The argument, experiment and findings: [`report/report.md`](report/report.md).
The same run narrated with outputs inline: [`notebooks/latent_message.ipynb`](notebooks/latent_message.ipynb).

## Reproduce

```bash
git clone https://github.com/SonghanWu/latent-message-handoff.git
cd latent-message-handoff
pip install -r requirements.txt

# correctness checks -- no download, no GPU, ~1 minute
python scripts/smoke_test.py
python scripts/integration_test.py

# the experiment -- ~20 min on a T4, downloads Qwen2.5-0.5B-Instruct and HotpotQA
python scripts/run_experiment.py --n-examples 100 --out results
```

Writes `rows.json`, `comparisons.csv`, `interaction.csv`, `main.png`, `scores.png`,
`reference_tensors.pt` and `score_tensors.pt` into `results/`. The copies already in
this repo are from the run reported in `report/report.md`.

## Layout

```
latent_comm/   kvcache · model_io · data · selection · experiment · stats
scripts/       smoke_test · integration_test · run_experiment
notebooks/     latent_message.ipynb
report/        report.md
results/       figures, tables, saved tensors
```
