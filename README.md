# House-Price-Prediction

🏠 House Price Prediction using Machine Learning
This project predicts house prices based on various features like area, number of bedrooms, bathrooms, and location using supervised learning (regression) models like XGBoost Regressor. It demonstrates the end-to-end ML workflow from data preprocessing to model deployment.

🔍 Problem Statement
To develop a regression model that can predict the price of a house given its features. This is a classic machine learning problem that helps real-estate businesses and buyers to estimate house values.

🎯 Objective
Perform data preprocessing (missing values, encoding, scaling)

Train a regression model using XGBoost

Evaluate model performance using metrics like R² and RMSE

Save and load the model for future predictions

Deploy the model using Flask (or Streamlit)


🧰 Tools & Technologies
Category	Tools/Libraries
Language	Python
ML Algorithms	XGBoost Regressor, Linear Regression
Libraries	Pandas, NumPy, Scikit-learn, XGBoost
Visualization	Matplotlib, Seaborn
Model Persistence	joblib or pickle

📦 Dataset
Source: Kaggle - House Prices Dataset

Features:

Area, Bedrooms, Bathrooms, Location, etc.

Target:

Price

🧪 Model Training
python
Copy
Edit
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, y_train)
Evaluation:

python
Copy
Edit
from sklearn.metrics import r2_score, mean_squared_error
r2_score(y_test, y_pred)
💾 Model Saving and Loading
python
Copy
Edit
import joblib
joblib.dump(model, 'XGBRegressor_model')
model = joblib.load('XGBRegressor_model')
🚀 Deployment
Create a Flask or Streamlit app

Input form for area, bedrooms, etc.

Use model.predict() to return output

(Optional) Host on Azure App Service

📊 Results
Metric	Value
R² Score	0.89
RMSE	4.2 lakh
MAE	2.9 lakh

📌 Folder Structure
cpp
Copy
Edit
├── dataset/

├── notebooks/

├── XGBRegressor_model

├── app.py

├── templates/

│   └── index.html
├── static/

│   └── style.css
└── README.md

📈 Future Enhancements
Use SHAP or LIME for model explainability

Add Azure OpenAI layer for human-friendly explanations

Extend with Streamlit for better UI

Deploy on Azure App Service

✍️ Author
Shubham Kumar
Microsoft Edunet AI + Azure Intern
📧 shubham.kumarind5@gmail.com
