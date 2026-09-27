"""
Publish trained model to Hugging Face Hub.

Required GitHub Secret:
    HF_TOKEN

Optional Environment Variables:
    HF_REPO_ID   (e.g. kinola-IQ/loan-risk-model)
"""

import os
from pathlib import Path

from huggingface_hub import HfApi


MODEL_FILE = "model_params.pkl"


def main() -> None:
    token = os.environ["HF_TOKEN"]

    repo_id = os.getenv(
        "HF_REPO_ID",
        "kinola-IQ/test-model",
    )

    model_path = Path(MODEL_FILE)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_FILE}"
        )

    api = HfApi(token=token)

    # Create repo if it doesn't already exist
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
    )

    # Upload model artifact
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=MODEL_FILE,
        repo_id=repo_id,
        repo_type="model",
    )

    print(
        f"Successfully published {MODEL_FILE} "
        f"to https://huggingface.co/{repo_id}"
    )


if __name__ == "__main__":
    main()