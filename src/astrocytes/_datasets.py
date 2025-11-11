"""OpenAstrocytes datasets"""

##
# Imports

from dataclasses import (
    dataclass,
    field,
)
from enum import Enum
import requests
import yaml

import astrocytes.schema as schema

import atdata
import toile.schema

from typing import (
    Type,
    Any,
)


##
# Constants

_DEFAULT_HIVE_ROOT = 'https://data.forecastbio.cloud/open-astrocytes'
_DEFAULT_MANIFEST_PATH = '/manifest.yml'


##
# General dataset information dataclass

@dataclass
class DatasetInfo:
    """TODO"""
    ##
    name: str
    """The OpenAstrocytes dataset identifier"""
    url: str
    """The WebDataset URL for this dataset"""
    sample_type: Type[atdata.PackableSample]
    """The sample type used for structuring this dataset"""

    # hive_root: str = '.'
    # """The root for the OA data hive"""

    # @property
    # def url( self ) -> str:
    #     """The full WebDataset URL specification for this dataset"""
    #     return self.hive_root + self.path
    
    @property
    def dataset( self ) -> atdata.Dataset:
        """TODO"""
        return atdata.Dataset[self.sample_type]( self.url )


##
# OA dataset layout
# TODO Rewrite w/ more flexible Pydantic validations

def _parse_dataset_info(
            config: dict[str, Any] | None,
            name: str,
            sample_type: Type[atdata.PackableSample],
            hive_root: str = '',
        ) -> DatasetInfo | None:
    
    if config is None:
        return None
    
    try:
        assert 'path' in config
        assert isinstance( config['path'], str )

        ret = DatasetInfo(
            name = name,
            url = hive_root + config['path'],
            sample_type = sample_type,
        )
    except:
        ret = None

    return ret

class GenericDatasetIndex:
    """TODO"""
    ##
    def __init__( self,
                config: dict[str, Any],
                hive_root: str = '',
            ):
        """TODO"""

        # Shortcut
        def _generic_info( name: str ) -> DatasetInfo | None:
            return _parse_dataset_info(
                config.get( name ),
                'generic/' + name,
                toile.schema.Frame,
                hive_root = hive_root,
            )

        self.bath_application = _generic_info( 'bath_application' )
        self.uncaging = _generic_info( 'uncaging' )

@dataclass
class TypedDatasetIndex:
    """TODO"""
    ##
    def __init__( self,
                config: dict[str, Any],
                hive_root: str = '',
            ):
        """TODO"""

        # Shortcut
        def _typed_info( name: str, sample_type: Type[atdata.PackableSample] )-> DatasetInfo | None:
            return _parse_dataset_info(
                config.get( name ),
                'typed/' + name,
                sample_type,
                hive_root = hive_root,
            )
        
        self.bath_application = _typed_info( 'bath_application', schema.BathApplicationFrame )
        self.uncaging = _typed_info( 'uncaging', schema.UncagingFrame )

@dataclass
class EmbeddingsDatasetIndex:
    """TODO"""
    ##
    def __init__( self,
                config: dict[str, Any],
                hive_root: str = '',
            ):
        """TODO"""

        # Shortcut
        def _typed_info( name: str, sample_type: Type[atdata.PackableSample] )-> DatasetInfo | None:
            return _parse_dataset_info(
                config.get( name ),
                'embeddings/' + name,
                sample_type,
                hive_root = hive_root,
            )
        
        self.bath_application = _typed_info( 'bath_application', schema.EmbeddingResult )
        # self.uncaging = _typed_info( 'uncaging', schema.UncagingFrame )

@dataclass
class PatchPCsDatasetIndex:
    """TODO"""
    ##
    def __init__( self,
                config: dict[str, Any],
                hive_root: str = '',
            ):
        """TODO"""

        # Shortcut
        def _typed_info( name: str, sample_type: Type[atdata.PackableSample] )-> DatasetInfo | None:
            return _parse_dataset_info(
                config.get( name ),
                'patch-pcs/' + name,
                sample_type,
                hive_root = hive_root,
            )
        
        self.bath_application = _typed_info( 'bath_application', schema.EmbeddingPCResult )
        # self.uncaging = _typed_info( 'uncaging', schema.UncagingFrame )

_EMPTY_CONFIG = {
    'bath_application': None,
    'uncaging': None,
}

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
                **_EMPTY_CONFIG,
                **config.get( 'generic', dict() )
            },
            hive_root = hive_root,
        )
        self.typed = TypedDatasetIndex(
            {
                **_EMPTY_CONFIG,
                **config.get( 'typed', dict() )
            },
            hive_root = hive_root,
        )
        self.embeddings = EmbeddingsDatasetIndex(
            {
                **_EMPTY_CONFIG,
                **config.get( 'embeddings', dict() )
            },
            hive_root = hive_root,
        )
        self.patch_pcs = PatchPCsDatasetIndex(
            {
                **_EMPTY_CONFIG,
                **config.get( 'patch_pcs', dict() )
            },
            hive_root = hive_root,
        )


##
# Data hive

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
            # Re-raise
            raise RuntimeError( f'Could not load OA manifest at {manifest_url}: {e}' )
        
        self.index = DatasetIndex( self._config,
            hive_root = self.root,
        )


#