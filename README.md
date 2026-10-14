Sales Prediction Using Machine Learning

A Machine Learning project to predict product sales based on advertising expenditure across different marketing channels.

1. Project Overview

Sales Prediction Using Machine Learning is a data science project that uses advertising expenditure data to estimate sales.

Businesses spend money on different advertising channels such as TV, Radio, and Newspaper. Understanding the relationship between advertising expenditure and sales can help businesses make better marketing and budgeting decisions.

In this project, machine learning models are trained using historical advertising data. The trained model is then used by a web application to predict sales based on user-provided advertising budgets.

The main objective is to understand how machine learning can be applied to a real-world regression problem.

2. Problem Statement

Businesses need to understand how advertising expenditure affects sales. Spending more on advertising does not always guarantee a proportional increase in sales.

This project addresses the problem by:

- Analyzing advertising expenditure and sales data.
- Identifying relationships between marketing channels and sales.
- Training machine learning models to predict sales.
- Comparing model performance using evaluation metrics.
- Providing an interactive interface for sales prediction.

3. Project Objectives

- Understand and analyze advertising data.
- Perform data exploration and preprocessing.
- Apply machine learning regression algorithms.
- Compare different models to identify the better-performing model.
- Evaluate predictions using standard regression metrics.
- Save the trained model for future use.
- Build an interactive application for making sales predictions.

4. Dataset Description

The project uses "Advertising.csv", which contains advertising expenditure and sales information.

The dataset includes the following variables:

Feature| Description
TV| Advertising expenditure on TV
Radio| Advertising expenditure on Radio
Newspaper| Advertising expenditure on Newspaper
Sales| Sales outcome used as the prediction target

The dataset contains 200 observations. The three advertising expenditure variables are used as input features, while Sales is the target variable.

5. Technologies Used

- Python: Main programming language.
- Pandas: Data loading and data manipulation.
- NumPy: Numerical computations.
- Scikit-learn: Model training, prediction and evaluation.
- Streamlit: Interactive web application.
- Matplotlib: Data visualization, where used.
- Jupyter Notebook: Exploratory data analysis.

6. Machine Learning Approach

This project treats sales prediction as a supervised learning regression problem.

Linear Regression

Linear Regression models the relationship between advertising expenditure and sales by fitting a linear equation to the training data.

It provides a useful baseline for understanding how changes in advertising budgets relate to predicted sales.

Random Forest Regressor

Random Forest Regressor combines predictions from multiple decision trees. It can learn more complex relationships between input features and the target variable.

Both models are evaluated to compare their predictive performance.

7. Project Workflow

1. Data Collection: Load the advertising dataset.
2. Data Exploration: Examine the dataset and understand the relationship between features and sales.
3. Data Preparation: Separate advertising expenditure features from the target variable.
4. Model Training: Train Linear Regression and Random Forest regression models.
5. Model Evaluation: Compare model performance using MAE, MSE, RMSE and R².
6. Cross-Validation: Use 5-fold cross-validation to assess model performance across different data splits.
7. Model Saving: Save the selected trained model as "sales_model.pkl".
8. Prediction: Load the saved model in the application and generate predictions from user inputs.

8. Model Evaluation

The project uses the following metrics to evaluate prediction quality:

- MAE (Mean Absolute Error): Measures the average absolute difference between actual and predicted sales.
- MSE (Mean Squared Error): Calculates the average squared prediction error.
- RMSE (Root Mean Squared Error): Expresses prediction error on the target's scale.
- R² Score: Measures how well the model explains variation in the target variable.
- 5-Fold Cross-Validation: Evaluates model performance across multiple data splits.

The training script compares the regression models and saves the selected model for use in the application.

9. Application Features

The Streamlit application provides an interactive way to use the trained model.

- Enter advertising budgets for TV, Radio and Newspaper.
- Generate a sales prediction using the saved model.
- View the entered advertising budget information.
- Explore a visual representation of the advertising expenditure.

The application connects the trained machine learning model with a simple user interface, making predictions easier to test without manually running model-training code each time.

10. Project Structure

sales-prediction-ml/
│
├── Advertising.csv
├── app.py
├── train_model.py
├── sales_model.pkl
├── explore.ipynb
├── .gitignore
└── README.md

- "Advertising.csv" — Dataset used for model training.
- "train_model.py" — Code for training and evaluating the models.
- "sales_model.pkl" — Saved trained model used for prediction.
- "app.py" — Streamlit application.
- "explore.ipynb" — Notebook for data exploration.
- ".gitignore" — Specifies files Git should ignore.
- "README.md" — Project documentation.

11. Installation and Setup

Prerequisites

- Python 3.9 or a compatible version.
- Git (optional, for cloning the repository).
- pip package manager.

Step 1: Clone the Repository

git clone https://github.com/gaurav16hub/sales-prediction-ml.git
cd sales-prediction-ml

Step 2: Install Dependencies

pip install pandas numpy scikit-learn streamlit matplotlib

Install any additional dependencies required by the project code.

Step 3: Train the Model (Optional)

python train_model.py

Run this step if you want to train the models again and regenerate the saved model.

Step 4: Run the Application

streamlit run app.py

The command starts the Streamlit application. Open the local URL displayed in the terminal to use the prediction interface.

12. Limitations

- Predictions depend on the quality and characteristics of the training dataset.
- The dataset is relatively small, with 200 observations.
- The model uses advertising expenditure features and does not account for every factor that may affect sales.
- Predictions should not be treated as guaranteed real-world sales outcomes.
- Model performance on unseen data may differ from training or cross-validation results.

13. Future Improvements

- Add more recent and diverse sales data.
- Improve the dashboard with additional visualizations.
- Include feature importance and model comparison charts.
- Deploy the application online for public access.
- Add more relevant business features to improve prediction quality.

14. Learning Outcomes

Through this project, I explored data analysis, regression algorithms, model evaluation, model serialization and the integration of a machine learning model into an interactive web application.

The project demonstrates an end-to-end machine learning workflow, from loading a dataset to generating predictions through a user interface.

15. Author

Gaurav Kumar

Computer Science Engineering Student

Maharaja Surajmal Institute of Technology, Delhi

GitHub: https://github.com/gaurav16hub

---

This project was developed for learning and demonstrating practical applications of machine learning and data analysis.
