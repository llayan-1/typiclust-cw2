# TPC_RP: Active Learning on a Budget — CIFAR-10 Implementation

**Coursework 2 | 5CCSAMLF Machine Learning | King's College London**

**Student:** Layan Alsubhi | K23065725

---
## Overview

This repository contains an implementation of the **TPC_RP** (Typical Clustering with Representation and Partition) active learning algorithm, based on the paper:

> Hacohen, G., Dekel, A., & Weinshall, D. (2022).
> *Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets.*
> Proceedings of the 39th ICML, PMLR 162.
> https://arxiv.org/abs/2202.02794

The algorithm is implemented on CIFAR-10 as part of Coursework 2 for 5CCSAMLF Machine Learning at King's College London.

---

## Algorithm Summary

TPC_RP operates in three steps:

1. **Representation Learning** — SimCLR self-supervised pre-training on the full unlabeled pool using a CIFAR-10 adapted ResNet-18 backbone
2. **Clustering for Diversity** — K-Means / Mini-Batch K-Means partitioning of the embedding space
3. **Typicality-Based Querying** — Selects the most typical (highest density) example from each uncovered cluster using K-NN distance

---

## Task 3: Proposed Modification

We propose a refined typicality estimator evaluated through an ablation study with four variants:

| Variant | Distance Metric | Neighbourhood Size |
|---|---|---|
| Original | Euclidean | Fixed K=20 |
| Variant A | Cosine | Fixed K=20 |
| Variant B | Euclidean | Adaptive K |
| Variant C | Cosine | Adaptive K |

**Justification:** SimCLR embeddings are L2-normalised, placing them on a unit hypersphere where cosine distance is geometrically more appropriate. Adaptive K produces more locally sensitive density estimates in small clusters, extending the paper's own use of min{20, cluster_size}.

---

## Results

| Round | Original | Variant A | Variant B | Variant C |
|---|---|---|---|---|
| 1 | 14.79% | 16.50% | 16.62% | 16.91% |
| 2 | 19.00% | 18.40% | 19.16% | 17.60% |
| 3 | 21.66% | 21.83% | 21.20% | 21.16% |
| 4 | 24.59% | 25.09% | 25.02% | 23.79% |
| 5 | 26.62% | 28.16% | 27.63% | 25.25% |

Variant A (cosine distance) achieves the most consistent improvement over the original. The combined Variant C shows non-additive behaviour, suggesting the two modifications interact in the density estimation step.

---

## Repository Structure

```
typiclust_cw2.ipynb    — Full implementation: Task 1 (original) + Task 3 (ablation study)
README.md              — This file
```

---

## Environment

- Python 3.10
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- Google Colab (T4 GPU recommended)

---

## References

- Hacohen et al. (2022) — https://arxiv.org/abs/2202.02794
- Chen et al. (2020) SimCLR — https://arxiv.org/abs/2002.05709
- Settles (2009) Active Learning Survey — http://burrsettles.com/pub/settles.activelearning.pdf
- Mittal et al. (2019) — https://arxiv.org/abs/1912.05361
