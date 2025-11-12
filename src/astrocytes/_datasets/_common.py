"""TODO"""

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
    """TODO"""
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
        """TODO"""
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
    """TODO"""
    ##
    def __init__( self,
                config: dict[str, Any],
                hive_root: str = '',
            ):
        """TODO"""

        print( 'hello!' )

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
    """Base for conversion from generic `toile` dataset Frame"""

    @staticmethod
    @abstractmethod
    def from_generic( s: Frame ) -> 'ExperimentFrame':
        """Convert a generic Frame to this specific kind of Frame"""
        pass


#