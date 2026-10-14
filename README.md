# Crowdfunding Success Prediction Using Classification Algorithms

## Project Overview

This project explores a crowdfunding dataset and builds machine-learning classification models to predict whether a campaign is successful. The target variable is `IsSuccessful`.

The notebook covers data exploration and preprocessing, feature selection, classification model training, cross-validation, hyperparameter tuning, and model comparison.

## Dataset

- **Source:** [crowdfunding.csv on GitHub](https://raw.githubusercontent.com/Srinithi-python/classification-algorithms-New/refs/heads/main/crowdfunding.csv)
- **Target column:** `IsSuccessful`
- The notebook removes the `Unnamed: 0` column when it exists.
- Categorical columns handled in the notebook include `VideoIncluded`, `LaunchMonth`, `Currency`, `Country`, and `Category`.

The notebook uses numerical features for the classification models. It does not include fixed performance results in this README because the reported metrics depend on running the notebook.

## Project Objectives

- Explore the dataset structure, columns, data types, and missing values.
- Preprocess data and encode categorical values.
- Select informative features using `SelectKBest` and `mutual_info_classif`.
- Split data into training and testing sets.
- Train and evaluate multiple classification algorithms.
- Apply cross-validation to assess model stability.
- Tune model hyperparameters using `GridSearchCV`.
- Compare model performance and identify a best-performing candidate.

## Workflow

1. **Data exploration**
   - Load the CSV dataset with pandas.
   - Inspect the first and last rows, shape, columns, data types, and missing values.
   - Drop `Unnamed: 0` if present.

2. **Preprocessing**
   - Encode selected categorical columns using `LabelEncoder`.
   - Keep numerical predictors for the classification workflow.
   - Replace infinite values with missing values and impute missing numerical values using the median.
   - Remove rows where `IsSuccessful` is missing and encode the target labels.

3. **Feature selection and splitting**
   - Use `SelectKBest` with mutual information to select up to five features.
   - Split the data into training and testing subsets, typically using an 80/20 split.

4. **Classification models**
   - Random Forest
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Gaussian Naive Bayes
   - Support Vector Machine (SVM)

5. **Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix
   - Training versus testing accuracy and their gap

6. **Cross-validation and hyperparameter tuning**
   - Use five-fold cross-validation to assess performance stability.
   - Use `GridSearchCV` to search selected hyperparameter combinations.
   - Use weighted F1-score for the multi-model tuning workflow.

7. **Model comparison**
   - Compare the tuned models on the test set.
   - Identify the model with the highest test weighted F1-score.

## Requirements

Install the main Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## How to Run

1. Clone or download this repository.
2. Ensure the dataset URL is accessible, or download `crowdfunding.csv` and update the notebook's data-loading path.
3. Open `Assignment_ClassificationAlgorithm.ipynb` in Jupyter Notebook, JupyterLab, Google Colab, or VS Code.
4. Run the cells in order from top to bottom.
5. Review the printed evaluation metrics and model-comparison results.

## Notes and Recommendations

- Run notebook cells in order so variables such as `df`, `X_train`, `X_test`, `y_train`, `y_test`, and `best_models` are defined before they are used.
- For a fair final evaluation, keep the test set separate from hyperparameter tuning and use it only for final model comparison.
- Fit imputation and scaling steps within a scikit-learn `Pipeline` during cross-validation to avoid data leakage.
- The notebook contains both feature-selection/scaling demonstrations and model-training sections. Ensure the same feature set is used consistently when comparing models.
- Label encoding categorical features can imply an artificial ordering. Consider one-hot encoding nominal features such as `Country`, `Currency`, and `Category` when appropriate.
- Use stratified splitting and `StratifiedKFold` for classification tasks, particularly when target classes are imbalanced.
- Select the final model using multiple metrics, especially weighted F1-score when class distribution is uneven, rather than relying on accuracy alone.

## Repository Contents

- `Assignment_ClassificationAlgorithm.ipynb` — notebook containing the analysis, model training, tuning, and comparison.
- `README.md` — project description and instructions.

## Author

Add your name or team details here.
