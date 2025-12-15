# [WQF7002] AI Techniques: Retinal OCT Prediction

This is a simple Streamlit application to detect retinal diseases using images from Optical Coherence Tomography (OCT) device.

Original dataset: [Retinal OCT images](https://www.kaggle.com/datasets/fabrizioravelli/retinal-oct-images-splitted)

## Prerequisites

- [uv](https://docs.astral.sh/uv)
- [make](https://www.geeksforgeeks.org/linux-unix/linux-make-command/)
    - On Linux and Mac, it should be installed by default.
    - On Windows, you need to install Scoop or Chocolatey, then install `make`. Or, use WSL.
- [git-lfs](https://git-lfs.com/): this is to download and upload large files to Git.
    - On Ubuntu: `sudo apt install git-lfs`
    - On Mac: (using Homebrew) `brew install git-lfs`
    - On Windows: Follow the installation instruction from `git-lfs` website.

## First-Time Setup

1) Run `uv sync`.
2) Run `git lfs install`.
3) Run `git lfs track "*.pt"`.

## Run

Run `make start`.
