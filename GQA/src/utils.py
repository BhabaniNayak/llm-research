import torch
import time
import psutil

from src.base import CustomPreTrainedModel


# Define a function to measure inference speed
def measure_inference_speed(model, tokenizer, input_text, num_runs=10):
    inputs = tokenizer(input_text, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_runs):
            if isinstance(model, CustomPreTrainedModel):
                outputs = model.model(**inputs)
            else:
                outputs = model(**inputs)
        end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    return avg_time


# Compare memory usage
def measure_memory_usage(model, tokenizer, input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt")

    # Move inputs to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)

    # Run the model to ensure it's loaded into memory
    model.eval()
    with torch.no_grad():
        if isinstance(model, CustomPreTrainedModel):
            outputs = model.model(**inputs)
        else:
            outputs = model(**inputs)

    if torch.cuda.is_available():
        memory_usage = torch.cuda.memory_allocated()
    else:
        process = psutil.Process()
        memory_usage = process.memory_info().rss  # in bytes

    return memory_usage
