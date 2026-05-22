from huggingface_hub import HfApi

api = HfApi()

repo_id = "archaia/dataset_v1"   

api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    private=True,
    exist_ok=True
)

api.upload_large_folder(
    folder_path="/data/user_data/udedhia/archaia/final",
    repo_id=repo_id,
    repo_type="dataset",
)

print("Upload complete.")