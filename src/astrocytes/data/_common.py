"""TODO"""

##
# Imports

import atdata

from abc import (
    ABC,
    abstractmethod,
)
from toile.schema import Frame
from typing import (
    Any,
    Self,
)


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