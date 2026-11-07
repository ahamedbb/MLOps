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

# Look for the CSV file anywhere locally
csv_paths = [
    "mlops/data/bank_customer_churn.csv",
    "data/bank_customer_churn.csv",
    "bank_customer_churn.csv",
    "../bank_customer_churn.csv"
]

target_csv = None
for path in csv_paths:
    if os.path.exists(path) and os.path.isfile(path):
        target_csv = path
        break

if target_csv:
    print(f"Found dataset at {target_csv}. Uploading file directly...")
    api.upload_file(
        path_or_fileobj=target_csv,
        path_in_repo="bank_customer_churn.csv",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Dataset file uploaded successfully.")
else:
    raise FileNotFoundError("Could not find bank_customer_churn.csv!")
