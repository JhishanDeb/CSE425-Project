import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.config import Config
from src.data_loader import MSDLoader, extract_segments
from src.features import TextFeatureExtractor, normalize_features
from src.utils import save_pickle, load_pickle, set_seed
from src.models import HybridVAE
from src.trainer import HybridVAETrainer
from src.analysis import compare_methods

def main():
    print("="*60)
    print("STARTING MEDIUM TASK (2500 SAMPLES) - HYBRID VAE & ADVANCED ANALYSIS")
    print("="*60)
    
    set_seed(42)
    config = Config(max_songs=2500, task_name="medium")
    
    # Paths
    audio_path = config.PROCESSED_DATA_DIR / 'audio_segments.npy'
    text_path  = config.PROCESSED_DATA_DIR / 'text_features.npy'
    ids_path   = config.PROCESSED_DATA_DIR / 'song_ids.pkl'
    
    # 1. Load Data & Extract Features
    if audio_path.exists() and text_path.exists() and ids_path.exists():
        print(f"Loading existing features from {config.PROCESSED_DATA_DIR}")
        audio_norm = np.load(audio_path) # (N, 100, 24)
        text_feats = np.load(text_path)  # (N, Vocab)
        song_ids   = load_pickle(ids_path)
    else:
        # Load Raw Data
        loader = MSDLoader(config.MSD_DIR)
        songs_data = loader.load_dataset(max_songs=config.MAX_SONGS)
        
        if not songs_data:
            print("No songs loaded. Exiting.")
            return

        # Extract Audio Segments (Timbre+Chroma)
        print("Extracting Audio Segments...")
        audio_segments, ids = extract_segments(songs_data, max_len=config.MAX_SEGMENTS) # (N, 100, 24)
        
        # Extract Text Features (Tags)
        print("Extracting Text Features (Tags)...")
        text_ext = TextFeatureExtractor(vocab_size=1000) # Smaller vocab for demo/speed
        text_feats, text_ids = text_ext.extract_from_songs(songs_data, text_key='tags')
        
        # Align datasets (in case some songs miss tags)
        # Note: extract_from_songs returns valid_ids. 
        # But we need to ensure audio and text correspond to same songs.
        # Simple intersection strategy:
        
        valid_set = set(text_ids)
        mask = [sid in valid_set for sid in ids]
        
        audio_segments = audio_segments[mask]
        song_ids = [sid for sid in ids if sid in valid_set]
        
        # Re-sort/align text feats to match audio_segments order if needed
        # But `extract_from_songs` might differ in order if it skips.
        # A safer way: iterate songs again or build dict.
        
        # Re-extraction with safety:
        # Let's filter songs_data first
        songs_with_tags = [s for s in songs_data if s.get('tags')]
        
        # Create fresh parallel lists
        final_audio = []
        final_tags = []
        final_ids = []
        
        print(f"Aligning Audio and Text (Found {len(songs_with_tags)} songs with tags)...")
        from tqdm import tqdm
        from src.data_loader import pad_or_truncate
        
        # Re-use the extractor fitted on all data? Or refit?
        # Let's refit on filtered data to be safe.
        text_ext = TextFeatureExtractor(vocab_size=1000)
        
        tag_strings = []
        for s in songs_with_tags:
            t = s['timbre']
            c = s['chroma']
            combined = np.hstack([t, c])
            padded = pad_or_truncate(combined, max_len=config.MAX_SEGMENTS)
            final_audio.append(padded)
            
            tags = s['tags']
            if isinstance(tags, list): tags = " ".join(tags)
            tag_strings.append(tags)
            
            final_ids.append(s['song_id'])
            
        final_audio = np.array(final_audio)
        text_ext.fit_vectorizer(tag_strings)
        final_text = text_ext.extract_features_batch(tag_strings) if hasattr(text_ext, 'extract_features_batch') else text_ext.vectorizer.transform(tag_strings).toarray()
        
        # Normalize Audio
        audio_norm, a_mean, a_std = normalize_features(final_audio)
        text_feats = final_text
        song_ids = final_ids
        
        # Save
        np.save(audio_path, audio_norm)
        np.save(text_path, text_feats)
        np.save(config.PROCESSED_DATA_DIR / 'audio_mean.npy', a_mean)
        np.save(config.PROCESSED_DATA_DIR / 'audio_std.npy', a_std)
        save_pickle(song_ids, ids_path)
        
    print(f"\nData Ready: Audio {audio_norm.shape}, Text {text_feats.shape}")
    
    # 2. Train HybridVAE
    print("\n" + "-"*30)
    print("2. TRAINING HYBRID VAE")
    print("-" * 30)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # DataLoaders
    n_train = int(0.8 * len(audio_norm))
    
    # Need to transpose audio to (N, C, L) for Conv1D
    # Current: (N, 100, 24) -> (N, 24, 100)
    X_audio = torch.FloatTensor(audio_norm).permute(0, 2, 1)
    X_text  = torch.FloatTensor(text_feats)
    
    train_ds = TensorDataset(X_audio[:n_train], X_text[:n_train])
    val_ds   = TensorDataset(X_audio[n_train:], X_text[n_train:])
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Configs
    audio_conf = {
        'input_channels': 24,
        'input_length': config.MAX_SEGMENTS, # 100
        'latent_dim': 32 # Internal to ConvVAE, but ignored since we take encoder output
    }
    text_conf = {
        'input_dim': text_feats.shape[1], # 1000
        'hidden_dim': 512
    }
    
    model = HybridVAE(audio_conf=audio_conf, text_conf=text_conf, latent_dim=config.LATENT_DIM, fusion_dim=64)
    
    trainer = HybridVAETrainer(model, train_loader, val_loader, config, device=device)
    save_path = config.MODELS_DIR / "hybrid_vae_best.pth"
    
    trainer.train(epochs=50, save_path=save_path)
    
    # Load Best
    if save_path.exists():
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {save_path}")

    # 3. Extract Latent
    print("\n" + "-"*30)
    print("3. EXTRACTING LATENT VECTORS")
    print("-" * 30)
    
    model.eval()
    all_loader = DataLoader(TensorDataset(X_audio, X_text), batch_size=config.BATCH_SIZE, shuffle=False)
    latent_vectors = []
    
    with torch.no_grad():
        for b_aud, b_txt in all_loader:
            mu, _ = model.encode(b_aud.to(device), b_txt.to(device))
            latent_vectors.append(mu.cpu().numpy())
            
    latent_vectors = np.concatenate(latent_vectors)
    np.save(config.VIZ_DIR / "hybrid_latent.npy", latent_vectors)
    
    # 4. Clustering & Analysis
    print("\n" + "-"*30)
    print("4. ADVANCED ANALYSIS & COMPARISON")
    print("-" * 30)
    
    # Baseline for comparison: Flattened Audio + Text
    # Raw features dim: (24*100) + 1000 = 3400
    flat_audio = audio_norm.reshape(len(audio_norm), -1)
    baseline_feats = np.hstack([flat_audio, text_feats])
    
    results = compare_methods(
        latent_vectors, 
        baseline_feats, 
        clustering_methods=['kmeans', 'agglomerative', 'dbscan'], 
        n_clusters=10, # More clusters for medium task
        save_dir=config.VIZ_DIR
    )
    
    print("\nPIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
