"""Core data discovery and access for OpenAstrocytes.

This module provides the main Hive class for discovering and accessing datasets
from the OpenAstrocytes data repository. The Hive fetches a manifest from the cloud,
parses dataset configurations, and provides organized access via dataset indices.

Main classes:
    - Hive: Entry point for accessing the data repository
    - DatasetIndex: Organized index of all available datasets
    - DatasetShortcuts: Convenience shortcuts for common dataset access patterns
"""

##
# Imports

# import atdata

import yaml
import requests

# from toile.schema import Frame
# from ._bath_application import BathApplicationFrame
# from ._uncaging import UncagingFrame
from ._embeddings import (
    # EmbeddingResult,
    # EmbeddingPCResult,
    EmbeddingsDatasetIndex,
    PatchPCsDatasetIndex,
)
    

from ._common import (
    # ST, # TypeVar for sample types
    # DatasetInfo,
    GenericDatasetIndex,
)

# from dataclasses import dataclass
from typing import (
    Any,
    # Type,
    # TypeVar,
)


##
# Constants for data repository layout

_DEFAULT_HIVE_ROOT = 'https://data.forecastbio.cloud/open-astrocytes'
_DEFAULT_MANIFEST_PATH = '/manifest.yml'

_EMPTY_EXPERIMENT_CONFIG = {
    'bath_application': None,
    'uncaging': None,
}


##
# Structured dataset info
# TODO Rewrite w/ more flexible Pydantic validations

class DatasetIndex:
    """Hierarchical index organizing all available datasets by type and experiment.

    Organizes datasets into three tiers:
        - generic: Raw Frame datasets (untyped imaging data)
        - embeddings: Vision transformer embedding outputs
        - patch_pcs: PCA-reduced patch embeddings

    Each tier is itself an index (GenericDatasetIndex, EmbeddingsDatasetIndex, etc.)
    providing access to experiment-specific datasets.

    Attributes:
        hive_root: Base URL for the data repository
        generic: Index of generic Frame datasets
        embeddings: Index of embedding result datasets
        patch_pcs: Index of PCA-reduced patch embedding datasets
    """
    ##

    def __init__( self,
                 config: dict[str, Any],
                 hive_root: str = '',
            ) -> None:
        """Initialize the dataset index from a configuration dictionary.

        Args:
            config: Parsed manifest configuration containing dataset paths
            hive_root: Base URL for the data repository
        """

        self.hive_root = hive_root

        # Build index
        self.generic = GenericDatasetIndex(
            {
                **_EMPTY_EXPERIMENT_CONFIG,
                **config.get( 'generic', dict() )
            },
            hive_root = hive_root,
        )
        # TODO Future
        # self.typed = TypedDatasetIndex(
        #     {
        #         **_EMPTY_EXPERIMENT_CONFIG,
        #         **config.get( 'typed', dict() )
        #     },
        #     hive_root = hive_root,
        # )
        self.embeddings = EmbeddingsDatasetIndex(
            {
                **_EMPTY_EXPERIMENT_CONFIG,
                **config.get( 'embeddings', dict() )
            },
            hive_root = hive_root,
        )
        self.patch_pcs = PatchPCsDatasetIndex(
            {
                **_EMPTY_EXPERIMENT_CONFIG,
                **config.get( 'patch_pcs', dict() )
            },
            hive_root = hive_root,
        )


##
# Main data hive class

class Hive:
    """Main entry point for discovering and accessing OpenAstrocytes datasets.

    The Hive fetches a YAML manifest from the data repository, parses dataset
    configurations, and creates a hierarchical DatasetIndex for organized access.
    By default, connects to the public OpenAstrocytes data repository.

    Attributes:
        root: Base URL for the data repository
        index: Hierarchical dataset index with organized access to all datasets

    Example:
        >>> hive = Hive()
        >>> # Access generic bath application frames
        >>> dataset = hive.index.generic.bath_application.dataset
        >>> # Access embeddings
        >>> embeddings = hive.index.embeddings.bath_application.dataset
    """

    def __init__( self,
                 root: str | None = None,
                 manifest_path: str | None = None,
            ) -> None:
        """Initialize the Hive and fetch the data manifest.

        Args:
            root: Base URL for the data repository. Defaults to the public
                OpenAstrocytes repository at data.forecastbio.cloud.
            manifest_path: Path to the manifest YAML file relative to root.
                Defaults to '/manifest.yml'.

        Raises:
            RuntimeError: If the manifest cannot be fetched or parsed
        """
        
        if root is None:
            root = _DEFAULT_HIVE_ROOT
        if manifest_path is None:
            manifest_path = _DEFAULT_MANIFEST_PATH
        
        self.root = root

        manifest_url = self.root + manifest_path
        try:
            response = requests.get( manifest_url )
            response.raise_for_status()

            manifest_text = response.text
            self._config = yaml.safe_load( manifest_text )

        except requests.exceptions.RequestException as e:
            # TODO Re-raise for now, rather than handling
            raise RuntimeError( f'Could not load OA manifest at {manifest_url}: {e}' )
        
        self.index = DatasetIndex( self._config,
            hive_root = self.root,
        )

class DatasetShortcuts:
    """Convenience shortcuts for accessing commonly-used datasets.

    Provides direct attribute access to frequently-used dataset combinations,
    avoiding the need to navigate the full Hive index hierarchy. Each attribute
    is either an atdata.Dataset or None if that dataset is unavailable.

    Attributes:
        bath_application: Generic bath application frames
        uncaging: Generic uncaging frames
        bath_application_embeddings: Bath application embedding results
        bath_application_patch_pcs: Bath application PCA-reduced embeddings

    Example:
        >>> from astrocytes import data
        >>> # Direct access instead of data._hive.index.generic.bath_application.dataset
        >>> dataset = data.bath_application
    """

    def __init__( self, hive: Hive ) -> None:
        """Initialize dataset shortcuts from a Hive instance.

        Args:
            hive: The Hive instance to create shortcuts for
        """
        ##

        self._hive = hive

        _ig = hive.index.generic
        self.bath_application = (
            _ig.bath_application.dataset if _ig.bath_application is not None
            else None
        )
        self.uncaging = (
            _ig.uncaging.dataset if _ig.uncaging is not None
            else None
        )

        _ie = hive.index.embeddings
        self.bath_application_embeddings = (
            _ie.bath_application.dataset if _ie.bath_application is not None
            else None
        )

        _ip = hive.index.patch_pcs
        self.bath_application_patch_pcs = (
            _ip.bath_application.dataset if _ip.bath_application is not None
            else None
        )

        # TODO more!

#