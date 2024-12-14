import os
from pathlib import Path
import argparse

from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import create_repo, login


def push_to_hub(
    hf_token: str,
    repo_id: str,
    finetune_result_path: str,
    commit_message: str = "Initial commit",
):
    login(hf_token)

    files = []

    for filename in os.listdir(finetune_result_path):
        file_path = os.path.join(finetune_result_path, filename)
        if os.path.isfile(file_path):
            files.append(file_path)

    model = CLIPModel.from_pretrained(finetune_result_path)
    processor = CLIPProcessor.from_pretrained(finetune_result_path)

    try:
        create_repo(repo_id=repo_id, repo_type="model")
    except Exception as e:
        print(e)
        print(f"Repository {repo_id} already exists. Skipping creation...")

    model.push_to_hub(
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial commit",
    )
    processor.push_to_hub(
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial commit",
    )

    print("Model and processor pushed to the hub successfully!")
    print(f"at: https://huggingface.co/{repo_id}")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API token")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID")
    parser.add_argument("--finetune_result_path", type=str, required=True, help="Path to the finetuned model")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    push_to_hub(args.hf_token, args.repo_id, args.finetune_result_path)