# Titanic Survival Analysis and Prediction 🚢

This project is an Exploratory Data Analysis (EDA) and Machine Learning study aimed at predicting the survival of Titanic passengers. It was developed as part of a Kaggle challenge and follows a step-by-step YouTube tutorial.

## Project Overview

The main goal of this project is to build a predictive model that estimates whether a passenger survived the Titanic disaster, based on historical data. The workflow includes:

* Exploratory Data Analysis (EDA): Understand the dataset, visualize distributions, and identify correlations between features.

* Data Preprocessing and Cleaning: Handle missing values, encode categorical variables, and remove irrelevant columns.

* Pipeline Building: Create a consistent processing pipeline for data preparation.

* Model Training: Train a `RandomForestClassifier` model to predict survival.

* Hyperparameter Optimization: Use `GridSearchCV` with cross-validation to find the best model parameters.

* Prediction Generation: Produce the final predictions for submission to Kaggle.

## Dataset

The dataset used is the "Titanic - Machine Learning from Disaster" challenge from Kaggle, available [here](https://www.kaggle.com/competitions/titanic/overview)

The project uses the `train.csv` (for training) and `test.csv`  (for prediction) files.

## Requirements

To run this notebook, you will need to have the following Python libraries installed. You can install them using `pip`:

```console
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

**1.** Clone this repository to your local machine.

**2.** Download `train.csv` and `test.csv` from Kaggle and place them in a folder named `datasets` in the project directory.

**3.** Open the notebook `titanic_analysis_survival_prediction.ipynb` in a Jupyter environment (Jupyter Notebook, JupyterLab, or VS Code).

**4.** Run the cells sequentially to reproduce the analysis and predictions.

## References & Credits

This project was inspired by the NeuralNine YouTube tutorial:

Tutorial Video: [Titanic Survival Prediction in Python - Machine Learning Project](https://www.youtube.com/watch?v=fATVVQfFyU0)

YouTube Channel: [NeuralNine](https://www.youtube.com/@NeuralNine)
