import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("loan_data.csv")

# Drop Loan_ID (not useful for prediction)
df.drop(columns=["Loan_ID"], inplace=True)

# Handle missing values
df.dropna(inplace=True)
df["Dependents"] = df["Dependents"].replace("3+", 3).astype(float)

# Encode categorical variables
encoder = LabelEncoder()
for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]:
    df[col] = encoder.fit_transform(df[col])

# Split data
x = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
#model.fit(X_train, y_train)
parameter={'n_estimators': np.arange(3,4,5),
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']}
CV=GridSearchCV(model,parameter,cv=5)
CV.fit(x_train,y_train)
best_model=CV.best_estimator_
# Save model
with open("loan_model.pkl", "wb") as f:
     pickle.dump(best_model, f)

