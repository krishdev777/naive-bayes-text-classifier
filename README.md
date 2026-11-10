# Probabilistic Text Classifier (Multinomial Naive Bayes)

A text classification engine implemented from scratch using Bayes' Theorem with Laplace Smoothing for spam detection.

## Performance
| Metric | Measurement | Description |
| :--- | :--- | :--- |
| **Training Latency** | ~35.0 microseconds | Vocabulary construction and class frequency mapping |
| **Inference Latency** | ~28.0 microseconds | Per-sample classification via log-likelihood sum |
| **Smoothing Method** | Add-1 (Laplace) | Prevents zero-probability underflow on unseen tokens |

## Formulation
Classification is determined via Maximum A Posteriori (MAP) estimation:
`P(C | W) proportional to P(C) * Product(P(w_i | C))`

Log-space transformations are applied to maintain numerical stability during likelihood products.

## Run
```bash
python3 classifier.py