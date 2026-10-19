# Synthetic data for burned-area segmentation

Code for the experiments in the master's thesis *Procedural Generation of
Synthetic Satellite Images for Wildfire Segmentation* (Nurbekuly, 2026). It
trains burned-area segmentation models on Sentinel-2 RGB tiles under 7-fold
geographic cross-validation and measures what procedurally generated synthetic
tiles add when real training data is scarce.

The real-data pipeline is a fork of the code released with *A Dataset for
Burned Area Delineation and Severity Estimation from Satellite Imagery*
(Colomba et al., CIKM 2022, [doi:10.1145/3511808.3557528](https://doi.org/10.1145/3511808.3557528)).

## Data

| Dataset | Source | Expected location |
|---|---|---|
| Satellite Burned Area Dataset (73 areas, 7 folds) | [doi:10.5281/zenodo.6597139](https://doi.org/10.5281/zenodo.6597139) | `sentinel-hub/<area>/` |
| synthetic-6000 (6,000 pre/post-fire scenes) | [doi:10.5281/zenodo.23117919](https://doi.org/10.5281/zenodo.23117919) | `synthetic-6000/<scene>/` |

Other locations can be passed with `--real-dir` and `--synthetic-dir`.
`satellite_data.csv` assigns each real area to one of the folds `coral`,
`cyan`, `grey`, `lime`, `magenta`, `pink` and `purple`.

Only the post-fire Sentinel-2 bands B04, B03 and B02 are used, cut into
256 x 256 tiles to match the synthetic renders.

## Installation

Python 3.11 and a CUDA GPU are assumed.

```bash
pip install -r requirements.txt
```

## Running

`cross_validation.py` runs one configuration; `python cross_validation.py --help`
lists every option. Each run writes to `logs/<run-name>/`:

- `report.csv`: one row per held-out fold with accuracy and per-class
  precision, recall and F1 (class 1 = burned). Burned-class IoU is
  `f1_1 / (2 - f1_1)`.
- `fold<i>_<fold>/`: the final model weights and a pickle with the confusion
  matrix, the train/test areas and the normalization statistics.

All runs evaluate the final-epoch model; there is no validation set and no
model selection. `--test-every N` only logs test-fold metrics during training.

Training regimes:

| Regime | Flags |
|---|---|
| Real only, step-matched to a mixed run | `--match-ratio 0.125` |
| Mixed: 1 synthetic tile in every 8 | `--synthetic-ratio 0.125` |
| Synthetic only, evaluated on every real fold | `--synth-only` |

`--real-frac f` keeps a fraction `f` of the real training areas, stratified per
fold and drawn from the seed and the held-out fold, so the step-matched and
mixed arms train on identical real areas at a given seed.

Each fold can be run on its own with `--test-folds <fold>`. Augmentation is
reseeded per fold, so a single-fold run reproduces that fold of a full 7-fold
run.

## Experiments

### Real-data scarcity grid

```
--model   {resnet_unet, segformer}
          {from scratch, --imagenet}
          {--match-ratio 0.125, --synthetic-ratio 0.125}
--real-frac {0.1, 0.25, 0.5, 1}
--test-folds {coral, cyan, grey, lime, magenta, pink, purple}
--seed {47, 48, 49}
--test-every 5
```

For example:

```bash
python cross_validation.py --model segformer --imagenet --real-frac 0.25 \
    --synthetic-ratio 0.125 --test-folds lime --seed 48 --test-every 5
```

### Synthetic-domain levers

Run against the scarcity grid at `--imagenet --real-frac 0.1 --synthetic-ratio 0.125`,
for both models, all seven folds and seeds 47-49:

| Lever | Flags |
|---|---|
| Histogram matching, fixed reference | `--hist-match` |
| Histogram matching, reference drawn per sample | `--hist-match --hm-stochastic --num-workers 4` |
| ... with background matched to real negatives | add `--hm-bg-negative` |
| Histogram + feature distribution matching | `--hist-match --fdm` |
| Feature distribution matching alone | `--fdm` |
| Per-domain loss split, equal weight | `--synth-loss-weight 1` |
| Domain-specific batch norm (`resnet_unet` only) | `--dsbn` |
| Pre-fire renders added as negatives | `--prefire-add` |

### Synthetic only

For both models, from scratch and with `--imagenet`:

```bash
python cross_validation.py --model resnet_unet --imagenet --synth-only \
    --epochs 5 --no-scheduler --test-every 1 --probe-every-steps 10
```

## Reproducibility

`--seed` seeds Python, NumPy and PyTorch, and cuDNN runs in deterministic
mode. Results can still differ slightly across GPUs and library versions. With
`--hm-stochastic`, the per-sample histogram-matching references depend on
`--num-workers`.

## Citation

If you use the real dataset, cite:

```bibtex
@inproceedings{10.1145/3511808.3557528,
    author = {Colomba, Luca and Farasin, Alessandro and Monaco, Simone and Greco, Salvatore and Garza, Paolo and Apiletti, Daniele and Baralis, Elena and Cerquitelli, Tania},
    title = {A Dataset for Burned Area Delineation and Severity Estimation from Satellite Imagery},
    year = {2022},
    publisher = {Association for Computing Machinery},
    doi = {10.1145/3511808.3557528},
    booktitle = {Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
    pages = {3893--3897},
    series = {CIKM '22}
}
```

## License

MIT, see [LICENSE](LICENSE).
