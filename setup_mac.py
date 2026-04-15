"""py2app bundler for fstt_mac.

Build dev bundle (fast, references venv — needs venv to stay in place):
    .venv/bin/python setup_mac.py py2app -A

Build standalone bundle (slow, self-contained):
    .venv/bin/python setup_mac.py py2app
"""
from setuptools import setup

APP = ["fstt_mac.py"]
DATA_FILES = ["prompt.md"]
OPTIONS = {
    "argv_emulation": False,
    "plist": {
        "CFBundleName": "fstt",
        "CFBundleDisplayName": "fstt",
        "CFBundleIdentifier": "com.fstt.mac",
        "CFBundleVersion": "0.1.0",
        "CFBundleShortVersionString": "0.1.0",
        "LSUIElement": True,
        "NSMicrophoneUsageDescription": "fstt records your voice for speech-to-text transcription.",
    },
    "packages": [
        "rumps", "pynput", "onnx_asr", "onnxruntime", "anthropic",
        "huggingface_hub", "pyperclip", "sounddevice", "numpy", "dotenv",
    ],
    "includes": ["AVFoundation", "ApplicationServices"],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
