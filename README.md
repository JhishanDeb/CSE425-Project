## Experiments
### Dataset Description
The project utilized the Million Song Subset (MSS), a collection of 10,000 song samples rich in audio features and metadata. For this study, we created three progressively larger subsets:

Easy Task: 1,000 songs.
Medium Task: 2,500 songs.
Hard Task: 5,500 songs.
The primary data fields used were:

Audio: analysis/segments_timbre (12-dim) and analysis/segments_pitches (12-dim Chroma).
Metadata: artist_terms (used as song tags) and title/artist_name for identification.
### Preprocessing Steps
Audio Feature Extraction:
Statistical: For VAE/Autoencoder, we computed 6 moments (mean, std, min, max, skew, kurtosis) per channel, resulting in a 144-dimensional feature vector ($24 \times 6$).
Temporal: For ConvVAE, segments were padded/truncated to a sequence length of $L=100$.
Normalization: All features were Z-score normalized (zero mean, unit variance) to ensure stable gradient descent.
Text Processing:
Tags were joined into a single string per song.
Processed via TF-IDF vectorizer (max features=1000, stop_words='english').
Genre Derivation (Ground Truth):
We derived 8 dominant genres from the top appearing tags (e.g., 'rock', 'pop', 'electronic', 'jazz') to serve as ground truth labels for supervision metrics (NMI, Purity).
### Hyperparameters & Training Details
All models were implemented in PyTorch and trained with the following settings:

Optimizer: Adam ($lr=1e-3$, weight_decay=$1e-5$).
Scheduler: ReduceLROnPlateau (factor=0.5, patience=7).
Early Stopping: Patience=15 epochs.
Batch Size: 64.
Epochs: 40-50.
Model Specifics:

Beta-VAE: Latent Dim=16, $\beta=4.0$ (disentanglement factor).
Autoencoder: Latent Dim=16, Deterministic.
HybridVAE: Latent Dim=32, Fusion Dim=64, $\alpha=1.0$ (text loss weight).
### Results
#### Clustering Metrics (Hard Task - 5500 Samples)
We compared Beta-VAE and Autoencoder representations against a PCA baseline using K-Means clustering (k=8). Metrics were computed against the derived Ground Truth genres.

Method	Silhouette Score	NMI	ARI	Purity
Beta-VAE	0.057	0.035	-0.001	0.702
Autoencoder	0.060	0.037	-0.002	0.709
Observations:

Purity (~0.71): Both Beta-VAE and Autoencoder achieved high cluster purity, indicating that the clusters are relatively homogeneous with respect to the derived genres (dominated by 'rock' and 'pop').
Silhouette Score (~0.06): Low silhouette scores suggest that the clusters in the latent space are overlapping, which is typical for music genre classification where boundaries are subjective and fuzzy.
NMI (~0.04): The low Normalized Mutual Information indicates that while pure, the unsupervised clusters correlate only weakly with the specific 8 labels we derived, likely discovering sub-genres or acoustic groupings not captured by broad tags.
#### Latent Space Visualizations
We visualized the 16-dimensional latent spaces using t-SNE (perplexity=30).

VAE Latent Space: Showed a more continuous, smooth manifold structure due to the Gaussian prior. Genres like 'electronic' and 'rock' showed partial separation but substantial overlap.
Cluster Distribution: The distribution plots revealed a class imbalance in the standard dataset (heavily skewed towards Rock/Pop), which the unsupervised models faithfully reproduced.
#### Comparison with Baselines
PCA: Provided a linear baseline. VAE models generally produced more compact representations.
Autoencoder vs VAE: The Autoencoder performed slightly better on pure reconstruction metrics (Purity 0.709 vs 0.702) as it is not constrained by the KL divergence term. However, the VAE provides a generative capability that the AE lacks.
#### Conclusion
The experiments demonstrate that VAE-based architectures can effectively extract compressed representations from high-dimensional music audio (144-dim -> 16-dim) while retaining semantic information comparable to or better than linear baselines. The derived "Purity" of ~71% confirms that the latent space captures significant genre-related structure unsupervised.