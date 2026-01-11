import os
from pathlib import Path

class Config:
    def __init__(self, max_songs=1000, task_name="default"):
        # === Robust way to get the project root directory ===
        # Assuming this is called from within the 'tasks/xxx/' or root, we want to normalize 
        # But since we are restructuring, let's assume we run from project root or find it relative to this file
        
        # This file is in src/config.py, so parent is src, parent.parent is root
        self.ROOT_DIR = Path(__file__).parent.parent.resolve()
        
        # ==================== PATHS ====================
        self.DATA_DIR = self.ROOT_DIR / 'data'
        
        # Find the actual MSD data directory dynamically as in the notebook
        self.MSD_DIR = self._find_msd_dir()
            
        self.RAW_DATA_DIR = self.DATA_DIR / 'raw'
        
        # Task specific output directory
        self.TASK_DIR = self.ROOT_DIR / 'results' / task_name
        self.PROCESSED_DATA_DIR = self.TASK_DIR / 'processed'
        self.MODELS_DIR = self.TASK_DIR / 'models'
        self.VIZ_DIR = self.TASK_DIR / 'viz'

        # Create directories
        for dir_path in [self.PROCESSED_DATA_DIR, self.MODELS_DIR, self.VIZ_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # ==================== HYPERPARAMS ====================
        self.MAX_SONGS = max_songs
        self.TRAIN_SPLIT = 0.7
        self.VAL_SPLIT = 0.15
        self.TEST_SPLIT = 0.15
        self.RANDOM_SEED = 42

        self.MAX_SEGMENTS = 100
        self.TIMBRE_DIM = 12
        self.CHROMA_DIM = 12
        self.AUDIO_FEATURE_DIM = 144

        self.MAX_LYRICS_LENGTH = 500
        self.VOCAB_SIZE = 5000
        self.LYRICS_EMBEDDING_DIM = 100
        
        # Training Params
        self.BATCH_SIZE = 64
        self.LATENT_DIM = 32
        self.HIDDEN_DIMS = [512, 256]
        self.DROPOUT = 0.2
        self.BETA = 4.0
        self.LR = 1e-3
        self.WEIGHT_DECAY = 1e-5
        self.EPOCHS = 200
        self.EARLY_STOPPING_PATIENCE = 15
        self.LR_SCHEDULER = {'factor': 0.5, 'patience': 7, 'min_lr': 1e-6}

        
        self.DEVICE = 'cuda'

        print(f"Config initialized for task: {task_name}")
        print(f"  > Project root: {self.ROOT_DIR}")
        print(f"  > Data source:  {self.MSD_DIR}")
        print(f"  > Output dir:   {self.TASK_DIR}")
        print(f"  > Max songs:    {self.MAX_SONGS}")

    def _find_msd_dir(self):
        """Locate the Million Song Subset directory"""
        base = self.DATA_DIR
        
        candidates = [
            Path("/Users/betopia/Downloads/MillionSongSubset"),
            base / "MillionSongSubset" / "A",
            base / "MillionSongSubset" / "B",
            base,
             Path("MyDrive/MillionSongSubset"),
             Path("/content/drive/MyDrive/MillionSongSubset"),
        ]
        
        for folder in candidates:
            if not folder.exists():
                continue
            
            # Check for h5 files
            # Just check if there is at least one to be fast
            try:
                if next(folder.rglob("*.h5"), None):
                    return folder
            except Exception:
                continue
                
        # Fallback to recursively searching from root if reasonable
        fallback = self.ROOT_DIR
        if next(fallback.rglob("*.h5"), None):
             return fallback
             
        print("WARNING: Could not auto-locate MSD .h5 files.")
        return base # Return default to avoid crashing immediately, though it will fail later
