import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
df = pd.read_csv("hf://datasets/ahamedb/bank-customer-churn/bank_customer_churn.csv")

X = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography']]
y = df['Exited']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
for name, data in [("Xtrain.csv", Xtrain), ("Xtest.csv", Xtest), ("ytrain.csv", ytrain), ("ytest.csv", ytest)]:
    data.to_csv(name, index=False)
    api.upload_file(path_or_fileobj=name, path_in_repo=name, repo_id="ahamedb/bank-customer-churn", repo_type="dataset")
print("Data preparation completed successfully.")
