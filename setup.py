from setuptools import setup, find_packages
import os
import subprocess

def install_packages():
    # Install required Python packages
    packages = [
        'deepspeed',
        'vinorm==2.0.7',
        'cutlet',
        'unidic==1.1.0',
        'underthesea',
        'gradio',
        'deepfilternet==0.5.6',
        'unidecode',
        'numpy',
        'huggingface-hub',
        'faster_whisper',
        'fasttext'
    ]
    for package in packages:
        subprocess.check_call([os.sys.executable, "-m", "pip", "install", package, "--use-deprecated=legacy-resolver"])

def clone_repository():
    # Clone the required Git repository
    if not os.path.exists('TTS'):
        subprocess.check_call(['git', 'clone', '--branch', 'add-vietnamese-xtts', 'https://github.com/thinhlpg/TTS.git'])
        subprocess.check_call([os.sys.executable, "-m", 'pip', 'install', '--use-deprecated=legacy-resolver', '-e', 'TTS'])

def download_model():
    # Assuming the snapshot_download function is available; otherwise, set up appropriately
    from huggingface_hub import snapshot_download
    print(" > Tải mô hình...")
    snapshot_download(repo_id="thinhlpg/viXTTS", repo_type="model", local_dir="model")

def install_unidic():
    # Install unidic
    subprocess.check_call([os.sys.executable, "-m", "unidic", "download"])

if __name__ == "__main__":
    clone_repository()
    install_packages()
    install_unidic()
    download_model()
    print(" > ✅ Cài đặt hoàn tất, bạn hãy chạy tiếp các bước tiếp theo nhé!")
