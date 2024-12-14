from pathlib import Path
import os

from loguru import logger


THIS_PATH = Path(__file__).resolve().parent
FINETUNE_RESULTS_ROOT = THIS_PATH.parent / "finetune_results"
MINIFIG_FINETUNE_ROOT = FINETUNE_RESULTS_ROOT / "minifigure_finetune"
BRICKS_FINETUNE_ROOT = FINETUNE_RESULTS_ROOT / "bricks_finetune"

HF_TOKEN = os.environ["HF_TOKEN"]

MINIFIG_CONFIG = {
    "repository_name": "clip-vit-base-patch32_lego-minifigure",
    "repo_id": "armaggheddon97/clip-vit-base-patch32_lego-minifigure",
    "finetune_py": str(THIS_PATH / "minifig_finetune.py"),
    "finetune_result_path": str(
        MINIFIG_FINETUNE_ROOT / "clip-vit-base-patch32_lego-minifigure"),
}
BRICKS_CONFIG = {
    "repository_name": "clip-vit-base-patch32_lego-bricks",
    "repo_id": "armaggheddon97/clip-vit-base-patch32_lego-bricks",
    "finetune_py": str(THIS_PATH / "bricks_finetune.py"),
    "finetune_result_path": str(
        BRICKS_FINETUNE_ROOT / "clip-vit-base-patch32_lego-bricks"),
}
PUSH_TO_HUB_PY = THIS_PATH / "push_to_hub.py"

def get_env_args():
    push_to_hub = os.environ["PUSH_TO_HUB"]
    if push_to_hub.lower() not in ["true", "false"]:
        raise ValueError("Invalid PUSH_TO_HUB value")
    finetune = os.environ["FINETUNE"]
    if finetune.lower() not in ["true", "false"]:
        raise ValueError("Invalid FINETUNE value")
    dataset = os.environ["DATASET"]
    if dataset not in ["minifigure", "bricks"]:
        raise ValueError("Invalid dataset. Available options: minifigure, bricks")
    
    return {
        "push_to_hub": True if push_to_hub.lower() == "true" else False,
        "finetune": True if finetune.lower() == "true" else False,
        "config": MINIFIG_CONFIG if dataset == "minifigure" else BRICKS_CONFIG,
    }

if __name__ == "__main__":
    configs = get_env_args()

    logger.info(
        f"Configs: push_to_hub={configs['push_to_hub']}, finetune={configs['finetune']}, dataset={configs['config']['repository_name']}"
    )

    if configs["finetune"]:
        # Run the finetuning script
        logger.info(f"Running finetuning for {configs['config']['repository_name']}")
        os.system(f"python3 {configs['config']['finetune_py']}")
        logger.success(f"Finetuning done for {configs['config']['repository_name']}")

    if configs["push_to_hub"]:
        # Push to the hub
        logger.info(f"Pushing to the hub for {configs['config']['repository_name']}")
        print(f"Pushing to the hub for {configs['config']['repository_name']}")
        push_to_hub_arg_str = (
            f"--hf_token={HF_TOKEN} "
            f"--repo_id={configs['config']['repo_id']} "
            f"--finetune_result_path={configs['config']['finetune_result_path']}"
        )
        os.system(f"python3 {PUSH_TO_HUB_PY} {push_to_hub_arg_str}")
        logger.success(f"Pushing to the hub done for {configs['config']['repository_name']}")
    
    logger.info("All done!")