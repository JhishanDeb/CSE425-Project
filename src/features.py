import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

class AudioFeatureExtractor:
    def __init__(self, feature_dim=144):
        self.feature_dim = feature_dim

    def _stats(self, arr):
        """mean, std, min, max, skew, kurtosis (12-dim each)"""
        if arr.shape[0] == 0:
            z = np.zeros(12)
            return [z]*6
        return [
            np.mean(arr, axis=0),
            np.std(arr, axis=0),
            np.min(arr, axis=0),
            np.max(arr, axis=0),
            skew(arr, axis=0),
            kurtosis(arr, axis=0)
        ]

    def extract_statistical_features(self, timbre, chroma):
        feats = []
        for data in [timbre, chroma]:
            feats.extend(self._stats(data))
        return np.concatenate(feats)   # 144-dim

    def extract_from_songs(self, songs_data):
        features, ids = [], []
        for song in tqdm(songs_data, desc="Audio feats"):
            vec = self.extract_statistical_features(song['timbre'], song['chroma'])
            features.append(vec)
            ids.append(song['song_id'])
        return np.array(features), ids

class TextFeatureExtractor:
    """
    Extract features from text (lyrics or tags) using TF-IDF.
    """
    def __init__(self, method='tfidf', vocab_size=5000):
        self.method = method
        self.vocab_size = vocab_size
        self.vectorizer = None

    def fit_vectorizer(self, text_list):
        if self.method != 'tfidf':
            raise ValueError("Only tfidf implemented here")
        self.vectorizer = TfidfVectorizer(
            max_features=self.vocab_size,
            max_df=0.8, min_df=5,
            stop_words='english'
        )
        self.vectorizer.fit(text_list)
        print(f"TF-IDF vocab size: {len(self.vectorizer.vocabulary_)}")

    def extract_features(self, text_item):
        if self.vectorizer is None:
            raise RuntimeError("Call fit_vectorizer first")
        return self.vectorizer.transform([text_item]).toarray()[0]

    def extract_from_songs(self, songs_data, text_key='tags'):
        """
        Extract text features from a list of song dictionaries.
        text_key: 'tags' (list of strings) or 'lyrics' (string)
        """
        text_list, valid_ids = [], []
        
        # Pre-process tags if needed (join list into string)
        processed_texts = []
        
        for song in songs_data:
            val = song.get(text_key)
            if not val:
                continue
                
            if isinstance(val, list):
                val = " ".join(val) # Join tags: ['rock', 'jazz'] -> "rock jazz"
            
            processed_texts.append(val)
            valid_ids.append(song['song_id'])

        if not processed_texts:
            print(f"No text data found for key: {text_key}")
            return None, []

        if self.vectorizer is None:
            self.fit_vectorizer(processed_texts)
            
        feats = self.vectorizer.transform(processed_texts).toarray()
        print(f"Text features shape: {feats.shape}")
        return feats, valid_ids

def normalize_features(X):
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std == 0] = 1
    return (X - mean) / std, mean, std
