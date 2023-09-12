import torch

def get_suitable_device(gb_vram_required):
    """
    Gets the index of the first GPU with enough free memory.

    Parameters:
    - gb_vram_required: Required GPU memory in GB.

    Returns:
    - "cuda:idx" if a suitable GPU is found, where idx is the GPU index.
    - None otherwise.
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        return None

    # Number of GPUs
    num_gpus = torch.cuda.device_count()

    # Convert required memory to bytes for comparison
    bytes_vram_required = gb_vram_required * 1024**3

    for idx in range(num_gpus):
        # Get total and free memory for the GPU
        total, free = torch.cuda.memory_stats(idx)["allocated_bytes.all.peak"], torch.cuda.memory_stats(idx)["reserved_bytes.all.peak"]
        if free >= bytes_vram_required:
            return f"cuda:{idx}"

    return None