from huggingface_hub import snapshot_download
import argparse

def download_folder_from_hf(
    repo_id,
    repo_folder,
    repo_type,
    local_dir,
    hf_token
):
    """
    Download entire folder from HuggingFace dataset (equivalent to upload_folder)
    """
    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=f"{repo_folder}/**",  # Download entire folder
            local_dir=local_dir,
            token=hf_token,
            local_dir_use_symlinks=False  # Get real files, not symlinks
        )
        
        print(f"🎉 Successfully downloaded {repo_folder}/ folder!")
        print(f"📁 Repository: https://huggingface.co/datasets/{repo_id}")
        print(f"📂 Files location: {local_dir}/{repo_folder}")
        print(f"📂 Downloaded to: {downloaded_path}")
        
        return downloaded_path
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def main(args):
    download_folder_from_hf(
        repo_id=args.repo_id,
        repo_folder=args.repo_folder,
        repo_type=args.repo_type,
        local_dir=args.local_dir,
        hf_token=args.hf_token
    )

# Usage - exact mirror of your upload code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download folder from HF repo")
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--repo_folder", type=str, required=True)
    parser.add_argument("--repo_type", type=str, default="dataset")
    parser.add_argument("--local_dir", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)

    args = parser.parse_args()
    main(args)


    