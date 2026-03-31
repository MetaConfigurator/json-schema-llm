from huggingface_hub import upload_folder

upload_folder(
    repo_id="durga-lakshmi-2000/qwen-2.5-coder-0.5B",
    folder_path="/content/drive/MyDrive/json_generation/deployment/qwen-0.5b-coder-merged",
    repo_type="model"
)
