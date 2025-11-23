"""Modal-based compute backends for OpenAstrocytes.

This package contains serverless compute functions that run on Modal cloud infrastructure:

    - embed: Image embedding using vision transformers (DINOv3)
    - pca: Incremental PCA for dimensionality reduction of embeddings

These backends handle GPU-accelerated and memory-intensive computations that
are impractical to run locally.
"""
