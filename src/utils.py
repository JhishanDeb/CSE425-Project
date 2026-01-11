import os
import pickle
import json
import numpy as np
import torch
import random
from pathlib import Path

def set_seed(seed=42):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_pickle(obj, filepath):
    """Save object as pickle file"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved: {filepath}")

def load_pickle(filepath):
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"Loaded: {filepath}")
    return obj

def save_json(obj, filepath):
    """Save object as JSON file"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=4)
    print(f"Saved: {filepath}")

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        obj = json.load(f)
    print(f"Loaded: {filepath}")
    return obj

def format_time(seconds):
    """Format seconds to readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def get_device():
    """Get available device (CUDA or CPU)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device
