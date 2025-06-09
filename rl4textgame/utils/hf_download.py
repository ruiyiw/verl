from huggingface_hub import snapshot_download
import os
import argparse


def main(args):
    # Download only specific folders to avoid rate limits
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        allow_patterns=[
            f"{args.hf_target_folder}/*",
        ],
        token=os.environ["HF_TOKEN"],
        repo_type=args.repo_type
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets/models from HF")
    parser.add_argument("--local_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--hf_target_folder", type=str, required=True)
    parser.add_argument("--repo_type", type=str, required=True)
    args = parser.parse_args()

    main(args)
   