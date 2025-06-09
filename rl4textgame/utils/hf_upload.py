from huggingface_hub import HfApi, create_repo
import argparse

def main(args):
    create_repo(args.repo_id, exist_ok=True, private=True)

    api = HfApi()
    api.upload_folder(
        folder_path=args.folder,
        path_in_repo=None,  
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        repo_type="model"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and upload models to HF")
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--path_in_repo", type=str, required=False)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--commit_message", type=str, default="Add checkpoints")
    args = parser.parse_args()

    main(args)
   