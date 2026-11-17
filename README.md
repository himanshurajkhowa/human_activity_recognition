# Robust Human Activity Recognition on Edge Devices

**ONGOING — Preliminary Research Release**

Can a phone tell walking from climbing stairs for someone the model has never seen? In my final-year CSE project, I use accelerometer and gyroscope recordings to study how well activity recognition carries over to new people and what still needs testing on a phone.

<img src="assets/pipeline.png" width="430" alt="Align accelerometer and gyroscope readings into usable windows. Predict with an SVM or neural network and score by participant.">

I prototyped the pipeline on UCI HAPT. The main experiments use seven activities from RealWorld HAR. I aligned the sensor streams by timestamp and kept track of missing inputs. The SVMs use features that summarize each window. The neural models read the windows directly.

## Development results

Each model was evaluated on nine people at familiar thigh and waist placements, with a separate fit that excluded each person in turn. The scores use windows with usable input and give each person equal weight. I reused this benchmark during development. Independent validation is still pending.

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| CNN-GRU C11 | 71.97% | 70.93% |
| B11Q RBF-SVM | 73.81% | 73.12% |
| D60 RBF-SVM | 76.60% | 76.19% |
| Attention-Residual BiGRU | 75.43% | 74.70% |

D60 remains the development comparator. Its lowest participant macro-F1 was **61.71%**. When scheduled predictions without usable input count as misses, its mean macro-F1 drops to **70.37%**. Walking and stairs still get confused, and some posture errors are concentrated in particular participants. B11Q and D60 share training rows and weights, so their comparison shows what changed when I added 12 lag features to F48 under the same SVM settings. Training conditions varied across the wider comparison. [Methods](docs/methods.md) · [Results and experiment history](docs/results.md)

## Run the public examples

Use Python 3.12. The first example aligns generated sensor streams, extracts features, fits a small SVM and scores a held synthetic group. The second redraws the D60 confusion matrix from saved aggregate counts. Exact benchmark replay requires the original research data, splits and model states.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m examples.synthetic_pipeline
python -m examples.plot_confusion
python -m unittest discover -s tests -v
```

## Current Status and Future Work

In EXP066, a CNN with parallel dilations scored above its matched control in five of nine source-validation choices. It passed the full set of criteria in zero choices and scored below D60 on the corresponding windows. D60 stays as the reference while I work on walking, stairs and posture errors.

B11Q ran in an earlier offline iPhone prototype. One D60 fold was ported to Swift, checked against offline fixtures and built as an unsigned iOS target. Live D60 operation and sustained phone performance still need testing. Separately annotated phone sessions will help measure how placement and orientation affect recognition.

[Dataset access and licensing](docs/data-and-licensing.md)
