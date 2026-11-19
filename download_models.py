
"""Download the project's local model assets.

Run from the repository root after installing requirements.txt.

Requirements:
    pip install -r requirements.txt

Notes:
    - Qwen is downloaded from Hugging Face.
    - YOLO11n is exported to NCNN if not already available.
    - SmolVLM is downloaded and managed by Ollama.
    - DeepSeek uses a cloud API and requires no local weights.
"""

from pathlib import Path
import subprocess
import sys
import shutil

from huggingface_hub import hf_hub_download, list_repo_files


# ==================================
# Project Root
# ==================================

ROOT = Path(__file__).resolve().parent


# ==================================
# 1. Download Qwen GGUF
# ==================================

def download_qwen():

    repo_id = "bartowski/Qwen_Qwen3-0.6B-GGUF"

    model_dir = ROOT / "models" / "Qwen"
    model_dir.mkdir(parents=True, exist_ok=True)

    target_path = model_dir / "Qwen3-0.6B-Q4_0.gguf"

    # Skip if model already exists
    if target_path.is_file() and target_path.stat().st_size > 0:
        print(f"Qwen already exists: {target_path}")
        return

    print("Checking available Qwen GGUF files...")

    files = list_repo_files(repo_id)

    matches = [
        f for f in files
        if f.lower().endswith(".gguf")
        and "q4_0" in f.lower()
    ]

    if not matches:
        raise RuntimeError(
            f"No Q4_0 GGUF file found in {repo_id}."
        )

    filename = matches[0]

    print(f"Downloading {filename}...")

    downloaded_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=model_dir,
        )
    )

    # Rename to the filename expected by llm.py
    if downloaded_path.resolve() != target_path.resolve():
        shutil.copy2(downloaded_path, target_path)

    print(f"Qwen ready: {target_path}")


# ==================================
# 2. Prepare YOLO11n NCNN
# ==================================

def prepare_yolo():

    target = (
        ROOT
        / "models"
        / "YOLO"
        / "yolo11n_ncnn_model"
    )

    # Skip if model already exists
    if target.is_dir() and any(target.iterdir()):
        print(f"YOLO NCNN already exists: {target}")
        return

    from ultralytics import YOLO

    print("Exporting YOLO11n to NCNN...")

    # Download YOLO11n weights if necessary
    model = YOLO("yolo11n.pt")

    # Export to NCNN format
    exported = Path(
        model.export(format="ncnn")
    )

    if not exported.is_absolute():
        exported = Path.cwd() / exported

    if exported.resolve() != target.resolve():

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copytree(
            exported,
            target,
            dirs_exist_ok=True,
        )

    print(f"YOLO NCNN ready: {target}")


# ==================================
# 3. Download SmolVLM via Ollama
# ==================================

def download_smolvlm():

    model = "ahmadwaqar/smolvlm2-256m-video:q8_0"

    print("Checking Ollama installation...")

    # Find Ollama executable
    ollama_path = shutil.which("ollama")

    # Windows fallback for Ollama executable
    if ollama_path is None and sys.platform == "win32":

        local_app_data = __import__("os").environ.get(
            "LOCALAPPDATA"
        )

        if local_app_data:
            candidate = (
                Path(local_app_data)
                / "Programs"
                / "Ollama"
                / "ollama.exe"
            )

            if candidate.is_file():
                ollama_path = str(candidate)

    if ollama_path is None:
        raise RuntimeError(
            "Ollama was not found. "
            "Install Ollama from https://ollama.com/download "
            "and restart the terminal."
        )

    print(f"Ollama found: {ollama_path}")

    # Check whether the model already exists
    result = subprocess.run(
        [ollama_path, "list"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )

    installed_models = [
        line.split()[0]
        for line in result.stdout.splitlines()[1:]
        if line.strip()
    ]

    if model in installed_models:
        print(f"SmolVLM already exists: {model}")
        return

    print(f"Pulling SmolVLM: {model}")

    subprocess.run(
        [ollama_path, "pull", model],
        check=True,
    )

    print(f"SmolVLM ready: {model}")


# ==================================
# 4. Main
# ==================================

def main():

    print("=" * 50)
    print("Downloading Local Models")
    print("=" * 50)

    tasks = [
        ("Qwen", download_qwen),
        ("YOLO", prepare_yolo),
        ("SmolVLM", download_smolvlm),
    ]

    for label, fn in tasks:

        print(f"\n[{label}]")

        try:
            fn()

        except Exception as exc:

            print(
                f"[FAILED] {label}: {exc}",
                file=sys.stderr,
            )

            sys.exit(1)

    print("\n" + "=" * 50)
    print("All local models are ready.")
    print("=" * 50)


if __name__ == "__main__":
    main()