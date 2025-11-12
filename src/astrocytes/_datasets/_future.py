"""(TODO) Interim work for future expansions of OA"""

##
# Imports

from ._common import (
    ST,
    DatasetInfo,
)
from ._bath_application import BathApplicationFrame
from ._uncaging import UncagingFrame

from dataclasses import dataclass
from typing import (
    Any,
    Type,
)


##
# Dataset indexes

@dataclass
class TypedDatasetIndex:
    """TODO For future directly-typed exports (that won't require lens conversion)"""
    ##
    def __init__( self,
                config: dict[str, Any],
                hive_root: str = '',
            ):
        """TODO"""

        # Shortcut
        def _typed_info( name: str, sample_type: Type[ST] )-> DatasetInfo[ST] | None:
            return DatasetInfo._parse(
                config.get( name ), 'typed/' + name,
                sample_type = sample_type,
                hive_root = hive_root,
            )
        
        self.bath_application = _typed_info( 'bath_application', BathApplicationFrame )
        self.uncaging = _typed_info( 'uncaging', UncagingFrame )


#