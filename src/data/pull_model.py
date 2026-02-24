from transformers import AutoModelForCausalLM


def pull_model(huggingface_tag: str):
    # Load model directly
    try:
        model = AutoModelForCausalLM.from_pretrained(huggingface_tag)
    except ConnectionError:
        print("Failed to load model")
        return
    return model