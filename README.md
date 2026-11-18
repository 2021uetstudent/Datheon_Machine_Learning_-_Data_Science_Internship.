# Supervised Classification on the Wine Dataset

A supervised machine-learning project that classifies wines into three cultivars from their chemical properties. It compares **Logistic Regression** and a **Decision Tree** and benchmarks them on accuracy, precision, recall and F1-score.

## Highlights

- Loads and inspects the Wine dataset with **Pandas** (shape, dtypes, summary statistics, null and duplicate checks)
- Exploratory analysis with **Seaborn** (pairplot of all features by class, class-distribution countplot)
- **80/20 stratified** train/test split
- Feature scaling with **`StandardScaler`**, fit on the training set only to avoid data leakage
- Trains **`LogisticRegression`** and **`DecisionTreeClassifier`**
- Evaluates with accuracy and macro-averaged precision, recall and F1-score
- **Confusion-matrix heatmaps** for both models
- **Decision-boundary plots** on a 2-D PCA projection of the scaled features
- Side-by-side benchmark table of both models

## Dataset

| Property | Value |
|---|---|
| Samples | 178 |
| Features | 13 numeric chemical measurements |
| Target | `class` (3 wine cultivars: 1, 2, 3) |
| Missing values | None |

Features: Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280/OD315 of diluted wines, Proline.

The data is the classic UCI Wine dataset, loaded from a local CSV (`Wine dataset.csv`).

## Methodology

1. **Load** the CSV into a Pandas DataFrame and check for nulls and duplicates.
2. **Explore** with a Seaborn pairplot coloured by class and a class-count plot.
3. **Split** into 142 training and 36 test samples (`test_size=0.2`, `random_state=42`, `stratify=y`), so each class keeps the same proportion in both sets.
4. **Scale** with `StandardScaler`: `fit_transform` on the training data, `transform` on the test data.
5. **Train** Logistic Regression (`max_iter=200`) on the scaled features and a Decision Tree classifier.
6. **Evaluate** on the held-out test set (macro averaging for precision, recall and F1).
7. **Visualise** confusion matrices and PCA-based decision boundaries.

## Results

Test-set performance (36 samples):

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.972 | 0.978 | 0.967 | 0.971 |
| Decision Tree | 0.972 | 0.978 | 0.972 | 0.974 |

Both models do very well on this dataset, and the Wine classes are largely separable. With only 36 test samples, one misclassified wine changes accuracy by about 2.8 points, so the gap between the two models is not meaningful. The Decision Tree has no fixed `random_state` in the notebook, so its numbers can vary slightly between runs.

> PCA with 2 components keeps about 55% of the variance, so the decision-boundary plots are a simplified view of the full 13-feature models.

## Project Structure

```
.
├── SupervisedClassificationModelonWinedataset.ipynb   # Full analysis notebook
├── Wine dataset.csv                                   # Dataset
└── README.md
```

## Getting Started

### Requirements

- Python 3.9+
- pandas, numpy, matplotlib, seaborn, scikit-learn (1.1+ for `DecisionBoundaryDisplay`), jupyter

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run

The notebook was written in Google Colab and reads the data from `/content/Wine dataset.csv`. To run it locally, change the path in the data-loading cell to:

```python
data = pd.read_csv("Wine dataset.csv")
```

Then launch Jupyter and run all cells:

```bash
jupyter notebook SupervisedClassificationModelonWinedataset.ipynb
```

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn

## Possible Improvements

- Add cross-validation and hyperparameter tuning (`GridSearchCV`) for more reliable estimates
- Tune the Decision Tree (`max_depth`, `min_samples_leaf`) and set `random_state` for reproducibility
- Add more models (Random Forest, SVM, k-NN) to the benchmark
- Inspect Logistic Regression coefficients and tree feature importances

## License

Released under the MIT License. See `LICENSE` for details.
