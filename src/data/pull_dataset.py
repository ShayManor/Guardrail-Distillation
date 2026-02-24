from datasets import load_dataset
import os


def pull_dataset(huggingface_tag: str):
    try:
        ds = load_dataset(huggingface_tag)
    except ConnectionError:
        print("Failed to download dataset")
        return

    print(f"Loaded dataset: {huggingface_tag}")
    return ds
