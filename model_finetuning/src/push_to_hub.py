import os
from pathlib import Path
import argparse

from loguru import logger
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
        logger.info(f"Repository {repo_id} already exists. Skipping creation...")

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

    logger.success("Model and processor pushed to the hub successfully!")
    logger.info(f"at: https://huggingface.co/{repo_id}")