"""Open astrocyte dynamics data.

This package provides a unified interface for discovering, loading, and processing
experimental imaging datasets from astrocyte neuroscience research.

The main entry points are:
    - `hive`: A global Hive instance for accessing the data repository manifest
    - `data`: Dataset shortcuts for convenient access to common datasets

Example:
    >>> import astrocytes
    >>> # Access bath application dataset
    >>> dataset = astrocytes.data.bath_application
    >>> # Access embeddings
    >>> embeddings = astrocytes.data.bath_application_embeddings
"""

##
# Imports

from ._datasets import (
    Hive,
    DatasetShortcuts,
)


##
# Expose

hive = Hive()
"""Global Hive instance for accessing the OpenAstrocytes data repository.

Automatically fetches and parses the manifest from the data repository on first access.
"""

data = DatasetShortcuts( hive )
"""Dataset shortcuts for convenient access to common datasets.

Provides direct access to:
    - `bath_application`: Generic bath application frames
    - `uncaging`: Generic uncaging frames
    - `bath_application_embeddings`: Bath application embedding results
    - `bath_application_patch_pcs`: Bath application PCA-reduced patch embeddings
"""


#