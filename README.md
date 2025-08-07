# ML Core

Cofounder led machine learning development and experimentation for internal systems. This repository serves as an R&D workspace for prototyping models and infrastructure that may support future production use.

## Purpose

This repo is intended for exploratory work in machine learning architecture, model design, and evaluation workflows. Components are iteratively developed with long-term system integration in mind.

## Highlights

- Experimental pipelines for supervised learning
- Modular model evaluation and tracking
- Reproducible development workflow
- Research-driven architecture, adaptable to evolving use cases

# Workflow
1. **Data Ingestion:** Load OHLCV data
2. **Feature Engineering:** Compute technical and statistical indicators
3. **Labeling:** Define classification target (e.g., `1` if future return > 0)
4. **Model Training:** Use XGBoost, logistic regression, etc.
5. **Prediction:** Apply model to unseen data
6. **Evaluation:** Accuracy, precision, recall, confusion matrix

## Tech Stack
- Python 3.10+
- pandas, numpy, scikit-learn, xgboost
- ta, matplotlib, seaborn

## Structure

```text
ml-core/
├── data/          # Input datasets and loaders
├── models/        # Training and inference logic
├── pipeline/      # Feature extraction and preprocessing
├── notebooks/     # Prototyping and experiments
├── utils/         # Reusable helpers
└── README.md
```

## Notes
- This is an internal module (NDA protected)
- No logic orexecution layer included

# Required Python Packages (requirements.txt)
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.6.0
ta
matplotlib
seaborn
jupyterlab


## Status

Early-stage development. Internal use only.  
Ownership: [Tom Le], Co founder

