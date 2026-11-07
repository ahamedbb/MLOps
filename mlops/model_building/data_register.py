import os
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo

repo_id = "ahamedb/bank-customer-churn"
repo_type = "dataset"
api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new repository...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

# Using upload_file directly to completely eliminate any folder path mismatch
csv_path = "mlops/data/bank_customer_churn.csv"
if not os.path.exists(csv_path):
    csv_path = "bank_customer_churn.csv"

print(f"Uploading file {csv_path} directly...")
api.upload_file(
    path_or_fileobj=csv_path,
    path_in_repo="bank_customer_churn.csv",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Dataset file uploaded successfully via upload_file.")
