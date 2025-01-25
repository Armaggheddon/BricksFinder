from pathlib import Path
import os

from loguru import logger

from finetune import finetune
from push_to_hub import push_to_hub

THIS_PATH = Path(__file__).resolve().parent
FINETUNE_RESULTS_ROOT = THIS_PATH.parent / "finetune_results"
MINIFIG_FINETUNE_ROOT = FINETUNE_RESULTS_ROOT / "minifigure_finetune"
BRICKS_FINETUNE_ROOT = FINETUNE_RESULTS_ROOT / "brick_finetune"


MINIFIG_CONFIG = {
    "repository_name": "clip-vit-base-patch32_lego-minifigure",
    "model_repo_id": "armaggheddon97/clip-vit-base-patch32_lego-minifigure",
    "dataset_id": "armaggheddon97/lego_minifigure_captions",
    "finetune_results_path": (
        MINIFIG_FINETUNE_ROOT / "clip-vit-base-patch32_lego-minifigure"
    ),
    "finetune_ckpt_path": MINIFIG_FINETUNE_ROOT / "checkpoints",
    "finetune_log_path": MINIFIG_FINETUNE_ROOT / "logs",
}
BRICKS_CONFIG = {
    "repository_name": "clip-vit-base-patch32_lego-brick",
    "model_repo_id": "armaggheddon97/clip-vit-base-patch32_lego-brick",
    "dataset_id": "armaggheddon97/lego_brick_captions",
    "finetune_results_path": (
        BRICKS_FINETUNE_ROOT / "clip-vit-base-patch32_lego-brick"
    ),
    "finetune_ckpt_path": BRICKS_FINETUNE_ROOT / "checkpoints",
    "finetune_log_path": BRICKS_FINETUNE_ROOT / "logs",
}

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
        "dataset": MINIFIG_CONFIG if dataset == "minifigure" else BRICKS_CONFIG,
    }

if __name__ == "__main__":
    configs = get_env_args()

    logger.info(
        "Arguments:\n"
        f"Push to Hub: {configs['push_to_hub']}\n"
        f"Finetune: {configs['finetune']}\n"
        f"Finetuning dataset: {configs['dataset']['repository_name']}"
    )

    if not FINETUNE_RESULTS_ROOT.exists():
        logger.error(
            f"Finetune results root directory not found at " 
            f"{FINETUNE_RESULTS_ROOT}.\nDid you forget to mount the volume?"
        )
        raise FileNotFoundError("Finetune results root directory not found")

    if configs["finetune"]:
        # Run the finetuning script
        logger.info(f"Running finetuning for {configs['config']['repository_name']}")
        finetune(
            dataset=configs["dataset"]["dataset_id"],
            finetune_results_path=configs["dataset"]["finetune_results_path"],
            finetune_ckpt_path=configs["dataset"]["finetune_ckpt_path"],
            finetune_log_path=configs["dataset"]["finetune_log_path"],
        )
        logger.success(f"Finetuning done for {configs['dataset']['repository_name']}")

    if configs["push_to_hub"]:
        # Push to the hub
        HF_TOKEN = os.environ.get("HF_TOKEN", "")
        print(HF_TOKEN)
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN not found in environment variables")

        logger.info(f"Pushing to the hub for {configs['dataset']['repository_name']}")
        push_to_hub(
            hf_token=HF_TOKEN,
            repo_id=configs["dataset"]["model_repo_id"],
            finetune_result_path=configs["dataset"]["finetune_results_path"],
        )
        logger.success(
            f"{configs['dataset']['repository_name']} has been pushed "
            f"to the hub at {configs['dataset']['model_repo_id']}"
        )
    
    logger.info("All done!")