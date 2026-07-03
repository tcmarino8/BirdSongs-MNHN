from pathlib import Path

from huggingface_hub import HfApi, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError

HF_USERNAME = "Tcmarino8"

api = HfApi()

roots = [
    Path(r"C:\Users\Salle-Cineradio\Documents\MachineLearning\BirdSongs-MNHN\Testing\DeepLabCut2\DavidBowie\Model519"),
    Path(r"C:\Users\Salle-Cineradio\Documents\MachineLearning\BirdSongs-MNHN\Testing\DeepLabCut2\Endive\Model519"),
]

for root in roots:
    if not root.exists():
        print(f"Skipping missing directory: {root}")
        continue

    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue

        # Prefix with bird name so repos are unique
        repo_name = f"{root.parent.name}-{model_dir.name}"
        repo_id = f"{HF_USERNAME}/{repo_name}"

        print(f"\nUploading {model_dir} -> {repo_id}")

        try:
            api.repo_info(repo_id, repo_type="model")
            print("  Repository already exists.")
        except RepositoryNotFoundError:
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=True,   # Change to True if desired
            )
            print("  Created repository.")

        upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
        )

        print("  Upload complete!")