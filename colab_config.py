import os
import gc
import torch

def setup_colab_environment():
    """Configure the environment for Google Colab"""
    # Set environment variables to handle frozen modules warning
    os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
    
    # Memory management settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    def clean_memory():
        """Clean up memory"""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return clean_memory

def optimize_model_loading(model_name, device='cpu'):
    """Optimize model loading settings"""
    return {
        'device_map': 'auto' if torch.cuda.is_available() else None,
        'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
        'low_cpu_mem_usage': True,
        'offload_folder': 'offload'
    }

# Create offload directory if it doesn't exist
if not os.path.exists('offload'):
    os.makedirs('offload')
