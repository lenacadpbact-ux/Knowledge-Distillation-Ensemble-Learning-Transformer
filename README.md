# Knowledge Distillation: Ensemble Learning & Transformer

> **Deep Learning Assignment 2** — Comparing ensemble-based and Transformer-based Knowledge Distillation for binary sentiment classification on the SST-2 dataset. All Deep Learning — no classical ML used.

---

## Overview

This project investigates **Knowledge Distillation (KD)** comparing two teacher paradigms:

- **Part A — Ensemble KD:** MLP + LSTM + TinyBERT teachers → Compact Student MLP
- **Part B — Transformer KD:** Pre-trained DistilBERT → Same Compact Student MLP

---

## Results

| Model | Type | Accuracy | vs Baseline |
|---|---|---|---|
| DistilBERT Teacher | Pre-trained Transformer | 91.06% | — |
| TinyBERT Teacher | Fine-tuned Transformer | 85.32% | — |
| Ensemble Teacher (Avg) | DL Combined | 76.95% | — |
| **Student — Transformer KD** | **Student** | **76.38%** | **+2.99% ✅** |
| **Student — Ensemble KD** | **Student** | **75.00%** | **+1.61% ✅** |
| MLP Teacher | DL Fully Connected | 74.43% | — |
| Baseline MLP (no KD) | Baseline | 73.39% | — |
| LSTM Teacher | DL Recurrent | 73.28% | — |

> Both students exceed the generic KD sentiment benchmark of < 2.33% reported by Zhang et al. (2025) — without GPU or LLM resources.

---

## Combined Results vs Literature

| Model | ACC (%) | Hinton et al. (2015) | Socher et al. (2013) | Sanh et al. (2019) | Mirzadeh et al. (2020) | Jiao et al. (2020) | Zhang et al. (2025) |
|---|---|---|---|---|---|---|---|
| Baseline MLP (No Distillation) | 72.94 | No KD applied | SST-2 dataset ✓ | — | — | — | — |
| MLP Teacher (DL Ensemble) | 74.43 | KD framework ✓ | SST-2 dataset ✓ | — | Small gap to student | — | — |
| LSTM Teacher (DL Ensemble) | 73.28 | KD framework ✓ | SST-2 dataset ✓ | — | Small gap to student | — | — |
| TinyBERT Teacher (DL Ensemble) | 85.55 | KD framework ✓ | SST-2 dataset ✓ | < 91.06% (DistilBERT) | Medium gap to student | Teacher role vs student (93.1%) ✓ | — |
| Ensemble Teacher (Avg of 3 DL) | 76.72 | Ensemble soft labels ✓ | SST-2 dataset ✓ | < 91.06% (DistilBERT) | Small gap to student | — | — |
| **Student — Ensemble KD** | **75.46** | Soft targets T scaling ✓ | SST-2 test 872 samples ✓ | — | Capacity gap confirmed ✓ | — | +2.52% > 2.33%  generic KD |
| DistilBERT Teacher (Pre-trained) | 91.06 | KD framework ✓ | SST-2 dataset ✓ | 91.06% ≈ 91.3% same model ✅ | Large gap to student | — | — |
| **Student — Transformer KD** | **76.15** | Soft targets T scaling ✓ | SST-2 test 872 samples ✓ | Distilled from DistilBERT ✓ | 25:1 gap 14.68% drop ✓ | — | +3.21% > 2.33%  ✅ |

---

## How to Run

1. Open `knowledge_distillation_sst2.ipynb` in **Google Colab**
2. Mount Google Drive — Step 1
3. Run all cells **top to bottom** — Steps 1 through 10
4. Results auto-save to `MyDrive/Assignment2/MLP_LSTM_DistilBERT/`

---

## Requirements

```bash
pip install datasets transformers torch tqdm matplotlib seaborn scikit-learn
```

---

## ⚙️ Key Hyperparameters

| Parameter | Part A (Ensemble) | Part B (Transformer) |
|---|---|---|
| Temperature T | 3.0 | 3.0 |
| Alpha α | 0.3 | 0.5 |
| Epochs | 30 | 30 |
| Optimizer | Adam lr=0.001 | Adam lr=0.001 |
| Batch size | 64 | 64 |

---

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015) Distilling the knowledge in a neural network. https://arxiv.org/pdf/1503.02531
- Jiao, X., et al. (2020) TinyBERT: Distilling BERT for NLU. EMNLP 2020. https://arxiv.org/pdf/1909.10351
- Minaee, S., et al. (2021) Deep learning-based text classification: A comprehensive review. ACM Computing Surveys, 54(3).
- Mirzadeh, S. I., et al. (2020) Improved knowledge distillation via teacher assistant. AAAI 2020. https://arxiv.org/pdf/1902.03393
- Nair, V., & Hinton, G. E. (2010) Rectified linear units improve restricted Boltzmann machines. ICML 2010.
- Sanh, V., et al. (2019) DistilBERT, a distilled version of BERT. https://arxiv.org/pdf/1910.01108
- Socher, R., et al. (2013) Recursive deep models for sentiment treebank. EMNLP 2013. https://aclanthology.org/D13-1170.pdf
- Zhang, Y., et al. (2025) Targeted distillation for sentiment analysis. EMNLP 2025. https://arxiv.org/pdf/2503.03225

---

*Deep Learning Assignment 2 · June 2026 · Google Colab CPU · SST-2 Sentiment Analysis*
