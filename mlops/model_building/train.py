import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

api = HfApi(token=os.getenv("HF_TOKEN"))

Xtrain = pd.read_csv("hf://datasets/ahamedb/bank-customer-churn/Xtrain.csv")
Xtest = pd.read_csv("hf://datasets/ahamedb/bank-customer-churn/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/ahamedb/bank-customer-churn/ytrain.csv")
ytest = pd.read_csv("hf://datasets/ahamedb/bank-customer-churn/ytest.csv")

numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
categorical_features = ['Geography']

class_weight = ytrain.value_counts().iloc[0] / ytrain.value_counts().iloc[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)
grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1)
grid_search.fit(Xtrain, ytrain.values.ravel())

best_model = grid_search.best_estimator_
joblib.dump(best_model, "best_churn_model.joblib")

repo_id = "ahamedb/churn-model"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

api.upload_file(
    path_or_fileobj="best_churn_model.joblib",
    path_in_repo="best_churn_model.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
