"""Common base classes and utilities for dataset management.

This module provides the foundational abstractions used across all dataset types:
    - DatasetInfo: Metadata and access for individual datasets
    - GenericDatasetIndex: Index for generic Frame datasets
    - ExperimentFrame: Abstract base class for typed experiment frames
"""

##
# Imports

import atdata

from dataclasses import dataclass
from abc import (
    ABC,
    abstractmethod,
)

from toile.schema import Frame

import typing
from typing import (
    Any,
    Type,
    TypeVar,
    Generic,
)

ST = TypeVar( 'ST', bound = atdata.PackableSample )
"""Type variable standing in for a packable sample type"""

##
# General dataset information dataclass

@dataclass
class DatasetInfo( Generic[ST] ):
    """Metadata and access information for a single dataset.

    Encapsulates dataset identification, location, and provides convenient access
    to the underlying atdata.Dataset instance. The generic type parameter ST
    specifies the sample type contained in the dataset.

    Attributes:
        name: Human-readable identifier for the dataset (e.g., 'generic/bath_application')
        url: Full WebDataset URL pointing to the TAR archive(s)
    """
    ##
    name: str
    """The OpenAstrocytes dataset identifier"""
    url: str
    """The WebDataset URL for this dataset"""
    # sample_type: Type[ST]
    # """The sample type used for structuring this dataset"""

    # hive_root: str = '.'
    # """The root for the OA data hive"""

    @property
    def sample_type( self ) -> type[ST]:
        """The type for each sample in this dataset"""
        # TODO Figure out why linting fails here
        return typing.get_args( self.__orig_class__ )[0]

    # @property
    # def url( self ) -> str:
    #     """The full WebDataset URL specification for this dataset"""
    #     return self.hive_root + self.path
    
    @property
    def dataset( self ) -> atdata.Dataset[ST]:
        """Create and return an atdata.Dataset instance for this dataset.

        Returns:
            A type-parameterized Dataset instance configured to load from this dataset's URL.
        """
        return atdata.Dataset[self.sample_type]( self.url )

    @classmethod
    def _parse(
                cls,
                config: dict[str, Any] | None,
                name: str,
                # TODO Would like to avoid this!
                sample_type: Type[ST],
                hive_root: str = '',
            ) -> 'DatasetInfo[ST] | None':
        """Parse dataset configuration into a DatasetInfo instance.

        Args:
            config: Configuration dictionary containing dataset metadata (must have 'path' key)
            name: Dataset identifier (e.g., 'generic/bath_application')
            sample_type: The type of samples in this dataset
            hive_root: Base URL for the data repository

        Returns:
            A DatasetInfo instance if parsing succeeds, None otherwise.
        """
        
        # TODO This is kind of a kludge
        # sample_type = typing.get_args( cls.__orig_bases__[0] )[0]
        # print( sample_type )

        if config is None:
            return None
        
        try:
            assert 'path' in config
            assert isinstance( config['path'], str )

            ret = DatasetInfo[sample_type](
                name = name,
                url = hive_root + config['path'],
            )
        except:
            ret = None

        return ret

class GenericDatasetIndex:
    """Index of available generic (untyped) Frame datasets.

    Provides access to raw imaging datasets as generic toile.Frame objects,
    before any experiment-specific type conversion. Each attribute (bath_application,
    uncaging, etc.) is either a DatasetInfo[Frame] or None if that dataset is unavailable.

    Attributes:
        bath_application: Generic frames from bath application experiments
        uncaging: Generic frames from photochemical uncaging experiments
    """
    ##
    def __init__( self,
                config: dict[str, Any],
                hive_root: str = '',
            ):
        """Initialize the generic dataset index.

        Args:
            config: Configuration dictionary mapping experiment types to dataset configs
            hive_root: Base URL for the data repository
        """

        # Shortcut
        def _generic_info( name: str ) -> DatasetInfo[Frame] | None:
            ret = DatasetInfo._parse(
                config.get( name ), 'generic/' + name,
                sample_type = Frame,
                hive_root = hive_root,
            )
            return ret

        self.bath_application = _generic_info( 'bath_application' )
        self.uncaging = _generic_info( 'uncaging' )


##
# Schema

## ABCs

class ExperimentFrame( ABC ):
    """Abstract base class for experiment-specific frame types.

    Defines the interface for converting generic toile.Frame objects into
    typed experiment frames (BathApplicationFrame, UncagingFrame, etc.).
    Subclasses must implement the from_generic() method to extract experiment-specific
    metadata and structure the data appropriately.
    """

    @staticmethod
    @abstractmethod
    def from_generic( s: Frame ) -> 'ExperimentFrame':
        """Convert a generic Frame to this specific experiment frame type.

        Args:
            s: A generic toile.Frame containing raw imaging data and metadata

        Returns:
            A typed experiment frame (subclass of ExperimentFrame) with extracted metadata
        """
        pass


#