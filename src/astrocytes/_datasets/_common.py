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

    # @property
    # def url( self ) -> str:
    #     """The full WebDataset URL specification for this dataset"""
    #     return self.hive_root + self.path
    
    @property
    def dataset( self ) -> atdata.Dataset[ST]:
        """TODO"""
        return atdata.Dataset[ST]( self.url )

    @classmethod
    def _parse(
                cls,
                config: dict[str, Any] | None,
                name: str,
                # sample_type: Type[ST],
                hive_root: str = '',
            ) -> 'DatasetInfo[ST] | None':
        
        if config is None:
            return None
        
        try:
            assert 'path' in config
            assert isinstance( config['path'], str )

            ret = DatasetInfo[ST](
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

        # Shortcut
        def _generic_info( name: str ) -> DatasetInfo[Frame] | None:
            return DatasetInfo[Frame]._parse(
                config.get( name ), 'generic/' + name,
                hive_root = hive_root,
            )

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