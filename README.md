# What Should a Latent Message Preserve?

DAPLab Latent-comm task. The write-up is [`report/report.md`](report/report.md); the run
with outputs inline is [`notebooks/latent_message.ipynb`](notebooks/latent_message.ipynb).

```bash
pip install -r requirements.txt

python scripts/smoke_test.py           # correctness checks, no download, no GPU
python scripts/integration_test.py

python scripts/run_experiment.py --n-examples 100 --out results   # ~20 min on a T4
```

`results/` holds the run reported in `report.md`.
