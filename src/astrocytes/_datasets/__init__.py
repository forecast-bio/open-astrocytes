"""TODO"""

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
    """TODO"""
    ##

    def __init__( self,
                 config: dict[str, Any],
                 hive_root: str = '',
            ) -> None:
        """TODO"""

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
    """TODO"""

    def __init__( self,
                 root: str | None = None,
                 manifest_path: str | None = None,
            ) -> None:
        
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
    """TODO"""

    def __init__( self, hive: Hive ) -> None:
        """TODO"""
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