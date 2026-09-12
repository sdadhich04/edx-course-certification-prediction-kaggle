# edX Course Certification Prediction

Coursework project for the University of Washington's CSE 416 (Introduction to Machine Learning). The notebook trains binary classifiers to predict the `certified` label in the supplied HarvardX/MITx Person-Course data.

## What it does

`edx_certification_prediction.ipynb`:

- loads the provided training and test CSV files;
- engineers engagement, activity, and age features;
- compares configurations of logistic regression, k-nearest neighbors, decision trees, random forests, AdaBoost, and an MLP neural network;
- tunes the decision threshold for the selected model; and
- retrains on the full training set and writes predictions to `submission.csv`.

The repository also includes the course column-description PDF and the generated submission file. The first notebook cell is an optional Google Colab download step; it can be skipped when the included CSV files are used.

## Tools and libraries

No specialized hardware is required. The project runs in a Python notebook environment and uses:

- Jupyter or Google Colab;
- pandas and NumPy;
- scikit-learn; and
- matplotlib and requests.

## Running

Install the listed Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Open `edx_certification_prediction.ipynb` from the repository in Jupyter or Google Colab and run the cells from top to bottom. Keep the included `edx_train.csv` and `edx_test.csv` in the working directory, and skip the optional download cell if they are already present. The final training cell writes `submission.csv`.

## Credits

- Sparsh Dadhich — notebook implementation, feature engineering, model comparison, threshold tuning, and written discussion.
- University of Washington CSE 416 — coursework assignment framework and the included course-provided data/resources.
