import h5py
from pathlib import Path
from tqdm import tqdm
import numpy as np

class MSDLoader:
    def __init__(self, path):
        self.path = Path(path)

    def get_h5_files(self, max_files=None):
        files = list(self.path.rglob("*.h5"))
        return files[:max_files] if max_files else files

    def load_song(self, h5_path):
        try:
            with h5py.File(h5_path, 'r') as f:
                def s(x):
                    v = f['metadata']['songs'][x][0]
                    return v.decode('utf-8') if isinstance(v, bytes) else str(v)

                year = int(f['musicbrainz']['songs']['year'][0])
                tags = [t.decode('utf-8') if isinstance(t, bytes) else str(t)
                        for t in f['metadata'].get('artist_terms', [])]

                return {
                    'song_id': s('song_id'),
                    'artist' : s('artist_name'),
                    'title'  : s('title'),
                    'year'   : year,
                    'timbre' : f['analysis']['segments_timbre'][:],
                    'chroma' : f['analysis']['segments_pitches'][:],
                    'tags'   : tags,
                    'filepath': str(h5_path)
                }
        except Exception as e:
            # print(f"Warning: {h5_path.name} → {e}")
            return None

    def load_dataset(self, max_songs=None):
        files = self.get_h5_files(max_songs)
        print(f"Loading {len(files)} songs …")
        songs = []
        for p in tqdm(files, desc="Loading songs", unit="song"):
            s = self.load_song(p)
            if s:
                songs.append(s)
        print(f"Loaded {len(songs)} songs.")
        return songs

def pad_or_truncate(array, max_len=100):
    """
    Pad or truncate features to a fixed length.
    Input shape: (seq_len, dim)
    Output shape: (max_len, dim)
    """
    if array.shape[0] >= max_len:
        return array[:max_len, :]
    else:
        padding = np.zeros((max_len - array.shape[0], array.shape[1]))
        return np.vstack([array, padding])

def extract_segments(songs_data, max_len=100):
    """
    Extracts padded segments from songs.
    Returns: numpy array of shape (N, max_len, dim) where dim = timbre + chroma
    """
    segments = []
    ids = []
    import numpy as np
    
    for song in tqdm(songs_data, desc="Processing segments"):
        t = song['timbre'] # (n, 12)
        c = song['chroma'] # (n, 12)
        
        # Combine timbre and chroma
        # shape: (n, 24)
        combined = np.hstack([t, c])
        
        padded = pad_or_truncate(combined, max_len=max_len)
        segments.append(padded)
        ids.append(song['song_id'])
        
    return np.array(segments), ids
