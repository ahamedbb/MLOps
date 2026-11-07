import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import joblib
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

api = HfApi(token=os.getenv("HF_TOKEN"))
Xtrain = pd.read_csv("hf://datasets/ahamedb/bank-customer-churn/Xtrain.csv")
ytrain = pd.read_csv("hf://datasets/ahamedb/bank-customer-churn/ytrain.csv")

numeric = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
categorical = ['Geography']

preprocessor = make_column_transformer((StandardScaler(), numeric), (OneHotEncoder(handle_unknown='ignore'), categorical))
model = make_pipeline(preprocessor, xgb.XGBClassifier(random_state=42))
model.fit(Xtrain, ytrain.values.ravel())

joblib.dump(model, "best_churn_model.joblib")
try:
    api.repo_info(repo_id="ahamedb/churn-model", repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id="ahamedb/churn-model", repo_type="model", private=False)

api.upload_file(path_or_fileobj="best_churn_model.joblib", path_in_repo="best_churn_model.joblib", repo_id="ahamedb/churn-model", repo_type="model")
print("Model trained and uploaded successfully.")
