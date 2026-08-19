# Predicting Course Certification: edX Learner Behavior (Kaggle Competition)

A binary classification model predicting whether an online-course learner
earned a certificate, trained on the HarvardX/MITx Person-Course dataset,
built as a class Kaggle competition entry — with real feature engineering,
a 12-configuration model comparison, decision-threshold tuning, and a
written discussion of the ethical risks of deploying a model like this.

Built from coursework for **CSE 416 (Introduction to Machine Learning)**,
University of Washington. See [Results](#results) for the model's actual
validation performance — not a target or expected value.

## The task

Predict the `certified` column of the HarvardX/MITx Person-Course AY2013
dataset — one row per learner per course, with features including event
counts (`nevents`), days active (`ndays_act`), video plays
(`nplay_video`), chapters viewed (`nchapters`), forum posts
(`nforum_posts`), year of birth, level of education, gender, and country.
8,758 labeled training rows, 2,920 unlabeled test rows to generate
predictions for.

## What this does

1. **Feature engineering** (`add_features`): derives per-day engagement
   rates from raw counts, applies `log1p` transforms to heavily
   right-skewed count features, adds binary "any activity" flags, and
   converts year-of-birth to age while explicitly handling a `-1`
   sentinel value as missing data rather than a real age.
2. **Model comparison**: trains and evaluates 12 model/hyperparameter
   configurations across 6 algorithm families — logistic regression,
   k-nearest neighbors, decision tree, random forest (2 configurations),
   AdaBoost, and a multi-layer perceptron — comparing validation accuracy
   across all of them.
3. **Threshold tuning**: rather than using the default 0.5 classification
   threshold, sweeps decision thresholds to find the one that maximizes
   validation accuracy for the winning model.
4. **Final submission**: retrains the winning configuration on the full
   training set and generates predictions for the test set.
5. **Written discussion**: three required discussion sections covering the
   feature-engineering process, a real debugging story, and the ethics of
   this kind of predictive model (see below).

## Results

- Best model: **random forest, 800 trees, unlimited depth** —
  **97.77%** validation accuracy, improved to **97.83%** after tuning the
  decision threshold to **0.51**.
- No official Kaggle leaderboard score is recorded in the notebook or
  supporting files — the 97.83% figure is the notebook's own held-out
  validation accuracy, not a leaderboard placement.
- `submission.csv` (included) is the actual generated prediction file for
  the 2,920-row test set, produced by the final retrained model.

## The written discussion (real, from the submission)

The notebook's discussion sections include a genuine debugging account —
a mismatch between 828 and 826 engineered features traced back to a
`reindex` step during feature alignment between train and test sets — and
a substantive ethics discussion: using a model like this to target
learners by country or education level for commercial purposes raises
real concerns, since the underlying data is observational (correlation,
not causation) and demographic features can act as proxies that make the
model's predictions — and any decisions based on them — discriminatory in
effect even without an explicit protected-class feature.

## What's original vs. course-provided

The dataset and the competition framework (train/test split, submission
format) were provided by the course. Everything else here is original
work: the entire feature-engineering pipeline, all 12 model
configurations and their comparison, the threshold-tuning step, and the
written discussion sections.

## Known limitations

- Two earlier, less-complete drafts of this notebook exist in the
  student's original files (one still carrying a placeholder team name and
  incomplete discussion sections) — this repo includes only the final,
  fully-completed version.
- No Kaggle leaderboard placement is available to cite — only the
  notebook's own validation accuracy.

## Running this code

Requires Python with `pandas`, `numpy`, and `scikit-learn`. Open
`edx_certification_prediction.ipynb` in Jupyter — `edx_train.csv`,
`edx_test.csv`, and `edX_column_description.pdf` (column reference) are
all included in this repo — and run all cells top to bottom.
