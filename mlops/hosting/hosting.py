from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
# If deployment folder doesn't exist yet, create a dummy app file so it doesn't fail
os.makedirs("mlops/deployment", exist_ok=True)
if not os.path.exists("mlops/deployment/app.py"):
    with open("mlops/deployment/app.py", "w") as app_file:
        app_file.write("import streamlit as st\nst.title('Bank Customer Churn Prediction')")

api.upload_folder(folder_path="mlops/deployment", repo_id="ahamedb/Bank-Customer-Churn", repo_type="space", path_in_repo="")
print("Space deployed successfully.")
