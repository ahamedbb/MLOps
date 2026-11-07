from huggingface_hub import HfApi
import os

repo_id = "ahamedb/Bank-Customer-Churn"
repo_type = "space"
api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="mlops/deployment",     
    repo_id=repo_id,          
    repo_type=repo_type,                      
    path_in_repo="",                          
)
