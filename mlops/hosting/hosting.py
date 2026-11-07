from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "ahamedb/Bank-Customer-Churn"

print(f"Pushing Streamlit deployment files to Space '{repo_id}'...")
api.upload_folder(
    folder_path="mlops/deployment",
    repo_id=repo_id,
    repo_type="space",
    path_in_repo=""
)
print("Space successfully updated with streamlit_app.py!")
