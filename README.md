# Age estimation based on human speech

This project focuses on building a regression pipeline to estimate the age of a speaker based on speech characteristics.
In particular, the proposed methodology is based on the segmentation of the dataset into two subgroups according to the length of the samples and on the extraction of Mel and MFCC features from audio recordings. 
Various regression models are compared to identify the most effective ones and perform the evaluation.

## Overview

Predicts speaker age (RMSE as evaluation metric) by combining metadata (pitch, jitter, shimmer, energy, gender, ethnicity, pauses, etc.) with audio features (Mel spectrogram, MFCC). The dataset is split into **Short samples** (brief sentences, no audio features extracted) and **Long samples** (longer, recurring sentences, audio features extracted via librosa). Gender/ethnicity interactions are added as engineered features. Four models are compared: Linear Regression, Ridge, Lasso, Random Forest, tuned with `GridSearchCV` (cv=5).

## Repository structure

```
.
├── project.ipynb     # Full pipeline: EDA, preprocessing, feature engineering,
│                      # audio feature extraction, model training & evaluation
├── report.pdf         # Full write-up of methodology and results
├── data/               # development.csv / evaluation.csv + audio files
└── audio_features/     # Cached extracted Mel/MFCC features
```

## Dataset

- 3,624 samples, 22,050 Hz audio: 2,933 labeled (development), 691 unlabeled (evaluation).
- Expected structure:
```
data/
├── development.csv
├── evaluation.csv
├── audios_development/
└── audios_evaluation/
```

## Results

| Subgroup | Best model | RMSE |
|---|---|---|
| Short samples | Random Forest | 4.349 |
| Long samples | Lasso (N_mel=60) | ~9.5 |
| Combined (public leaderboard) | RF + Lasso | 9.585 |

The dataset is imbalanced toward the 15–25 age range, causing overestimation there and underestimation for over-sixties.

## Requirements

```bash
pip install pandas numpy matplotlib seaborn librosa scikit-learn
```

## Usage

```bash
git clone https://github.com/soniafoco/Age-estimation-based-on-human-speech.git
cd Age-estimation-based-on-human-speech
jupyter notebook project.ipynb
```

Place the dataset in `data/` as described above before running.

## Report

See [`report.pdf`](./report.pdf) for full methodology, figures, and references.

## Authors

- Pasqualina Ambrosio – Politecnico di Torino (s345061)
- Sonia Foco – Politecnico di Torino (s343043)
