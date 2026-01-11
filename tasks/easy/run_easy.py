import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.config import Config
from src.data_loader import MSDLoader
from src.features import AudioFeatureExtractor, normalize_features
from src.utils import save_pickle, load_pickle, set_seed
from src.models import VAE
from src.trainer import VAETrainer
from src.analysis import compare_methods

def main():
    print("="*60)
    print("STARTING EASY TASK (1000 SAMPLES) - FULL PIPELINE")
    print("="*60)
    
    set_seed(42)
    
    # 1. Initialize Config
    # Check if we should use existing data to save time or re-run extracted
    config = Config(max_songs=1000, task_name="easy")
    
    # Check if features already exist
    audio_path = config.PROCESSED_DATA_DIR / 'audio_features.npy'
    ids_path = config.PROCESSED_DATA_DIR / 'song_ids.pkl'
    
    if audio_path.exists() and ids_path.exists():
        print(f"Loading existing features from {config.PROCESSED_DATA_DIR}")
        audio_norm = np.load(audio_path)
        song_ids = load_pickle(ids_path)
    else:
        # Load Data & Extract
        loader = MSDLoader(config.MSD_DIR)
        songs_data = loader.load_dataset(max_songs=config.MAX_SONGS)
        
        if not songs_data:
            print("No songs loaded. Exiting.")
            return

        audio_ext = AudioFeatureExtractor(feature_dim=config.AUDIO_FEATURE_DIM)
        audio_feats, song_ids = audio_ext.extract_from_songs(songs_data)
        
        # Normalize & Save
        audio_norm, a_mean, a_std = normalize_features(audio_feats)
        np.save(config.PROCESSED_DATA_DIR / 'audio_features.npy', audio_norm)
        np.save(config.PROCESSED_DATA_DIR / 'audio_mean.npy', a_mean)
        np.save(config.PROCESSED_DATA_DIR / 'audio_std.npy',  a_std)
        save_pickle(song_ids, config.PROCESSED_DATA_DIR / 'song_ids.pkl')
    
    print(f"\nFeatures ready: {audio_norm.shape}")
    
    # 2. Train VAE
    print("\n" + "-"*30)
    print("2. TRAINING VAE")
    print("-" * 30)
    
    # Prepare DataLoaders
    n_train = int(0.8 * len(audio_norm))
    X_train = torch.FloatTensor(audio_norm[:n_train])
    X_val = torch.FloatTensor(audio_norm[n_train:])
    
    train_loader = DataLoader(TensorDataset(X_train), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val),   batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Initialize Model & Trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VAE(input_dim=config.AUDIO_FEATURE_DIM, latent_dim=config.LATENT_DIM)
    
    trainer = VAETrainer(model, train_loader, val_loader, config, device=device)
    
    # Train
    save_path = config.MODELS_DIR / "vae_best.pth"
    trainer.train(epochs=50, save_path=save_path) # Reduced epochs for demo, configurable
    
    # Load Best Model
    if save_path.exists():
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {save_path}")
    
    # 3. Extract Latent Space
    print("\n" + "-"*30)
    print("3. EXTRACTING LATENT VECTORS")
    print("-" * 30)
    
    model.eval()
    all_loader = DataLoader(TensorDataset(torch.FloatTensor(audio_norm)), batch_size=config.BATCH_SIZE, shuffle=False)
    latent_vectors = []
    
    with torch.no_grad():
        for batch in all_loader:
            x = batch[0].to(device)
            mu, _ = model.encode(x)
            latent_vectors.append(mu.cpu().numpy())
            
    latent_vectors = np.concatenate(latent_vectors)
    print(f"Latent shape: {latent_vectors.shape}")
    
    np.save(config.VIZ_DIR / "latent_vectors.npy", latent_vectors)
    
    # 4. Analysis & Comparison
    print("\n" + "-"*30)
    print("4. ANALYSIS & VISUALIZATION")
    print("-" * 30)
    
    results = compare_methods(latent_vectors, audio_norm, n_clusters=5, save_dir=config.VIZ_DIR)
    
    print("\nPIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Outputs saved in:")
    print(f" - Features: {config.PROCESSED_DATA_DIR}")
    print(f" - Models:   {config.MODELS_DIR}")
    print(f" - Viz:      {config.VIZ_DIR}")

if __name__ == "__main__":
    main()
