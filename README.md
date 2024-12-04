# employee-attrition-prediction
A machine learning project for predicting employee attrition using Python.
Employee Attrition Prediction
Overview
This project focuses on predicting employee attrition using a machine learning model. By analyzing employee data, the project provides insights into factors that contribute to attrition, helping organizations take proactive measures to improve employee retention.

Dataset
The dataset used in this project includes the following features:

Age: Employee age in years.
DistanceFromHome: Distance between the employee's home and workplace (in miles).
MonthlyIncome: Employee's monthly income (in USD).
JobSatisfaction: Job satisfaction level (scale: 1–4).
Attrition: Target variable (1 = Attrited, 0 = Stayed).
The dataset is stored in the file employee_attrition_dataset.csv.

Methodology
Data Preprocessing:

Handled missing data and feature selection.
Normalized and split the dataset into training and testing subsets.
Model Selection:

Random Forest Classifier was chosen for its ability to handle small datasets and provide robust predictions.
The model was trained on 70% of the dataset and tested on the remaining 30%.
Evaluation:

Generated a confusion matrix to visualize performance.
Evaluated precision, recall, F1-score, and accuracy using a classification report.
Results
Classification Report: The model's precision, recall, and F1-score are summarized in classification_report.txt.
Confusion Matrix: Visual representation of predicted vs. actual results to assess performance.
Files in Repository
employee_attrition_dataset.csv: Dataset used for training and testing.
employee_attrition_code.py: Python script for data preprocessing, model training, and evaluation.
classification_report.txt: Detailed metrics evaluating the model.
How to Run the Project
Clone the repository:

bash
Copy code
git clone https://github.com/stephanie1032/employee-attrition-prediction
cd employee-attrition-prediction
Install dependencies:

bash
Copy code
pip install -r requirements.txt
(Include a requirements.txt file if needed, specifying libraries such as pandas, scikit-learn, matplotlib, and seaborn.)

Run the Python script:

bash
Copy code
python employee_attrition_code.py
Outputs:

Confusion matrix displayed as a visualization.
Classification report printed in the terminal and saved as classification_report.txt.
Future Enhancements
Add additional features such as tenure, performance ratings, and education level.
Experiment with advanced models like XGBoost and Gradient Boosting.
Develop a web-based dashboard for interactive visualizations and real-time predictions.

