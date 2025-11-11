"""OpenAstrocyte-specific dataset schemas"""

##
# Imports

import atdata
import toile.schema as ts

from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

from typing import (
    Literal,
    TypeAlias,
    Self,
    Any,
)
from numpy.typing import (
    NDArray,
)


##
# Sample types

## ABCs

class ExperimentFrame( ABC ):
    """Base for conversion from generic `toile` dataset Frame"""

    @staticmethod
    @abstractmethod
    def from_generic( s: ts.Frame ) -> Any:
        """Convert a generic Frame to this specific kind of Frame"""
        pass

##

@dataclass
class EmbeddingPCResult( atdata.PackableSample ):
    """TODO"""
    patch_pcs: NDArray
    #
    metadata: dict[str, Any] | None = None

@dataclass
class EmbeddingResult( atdata.PackableSample ):
    """TODO"""
    cls_embedding: NDArray
    registers: NDArray | None = None
    patches: NDArray | None = None
    #
    metadata: dict[str, Any] | None = None


## Bath application

BathApplicationCompound: TypeAlias = Literal[
    'baclofen',
    'tACPD',
    'unknown',
]

@dataclass
class BathApplicationFrame( atdata.PackableSample, ExperimentFrame ):
    """TODO"""
    ##

    applied_compound: BathApplicationCompound
    """TODO"""
    image: NDArray
    """TODO"""
    t_index: int
    """Frame index in the overall sequence of the original recording"""
    t: float
    """Time (in seconds) this frame was captured after the start of the original recording"""

    date_acquired: str | None = None
    """ISO timestamp at approximately when the experiment was performed"""

    mouse_id: str | None = None
    """Identifier of the mouse this slice was taken from"""
    slice_id: str | None = None
    """Identifier of the slice this recording was made from"""
    fov_id: str | None = None
    """Identifier of the field of view within an individual slice that was
    recorded
    """
    movie_uuid: str | None = None
    """OME UUID of the full tseries"""

    scale_x: float | None = None
    """The size of each pixel in the $x$-axis (in microns)"""
    scale_y: float | None = None
    """The size of each pixel in the $y$-axis (in microns)"""

    ## Specification lenses

    @staticmethod
    def from_generic( s: ts.Frame ) -> 'BathApplicationFrame':
        return _specify_bath_application( s )

# Register lenses

@atdata.lens
def _specify_bath_application( s: ts.Frame ) -> BathApplicationFrame:
    assert s.metadata is not None
    return BathApplicationFrame(
        # TODO
        applied_compound = 'unknown',
        image = s.image,
        t_index = s.metadata['frame']['t_index'],
        t = s.metadata['frame']['t'],
        #
        date_acquired = s.metadata['date_acquired'],
        movie_uuid = s.metadata['uuid'],
        #
        scale_x = s.metadata['scale_x'],
        scale_y = s.metadata['scale_y'],
    )

## NT Uncaging

UncagingCompound: TypeAlias = Literal[
    'gaba',
    'glu',
    'laser_only',
    'unknown',
]

@dataclass
class UncagingFrame( atdata.PackableSample ):
    """TODO"""
    ##

    uncaged_compound: UncagingCompound
    """TODO"""
    image: NDArray
    """TODO"""
    t_index: int
    """Frame index in the overall sequence of the original recording"""
    t: float
    """Time (in seconds) this frame was captured after the start of the original recording"""

    date_acquired: str | None = None
    """ISO timestamp at approximately when the experiment was performed"""

    mouse_id: str | None = None
    """Identifier of the mouse this slice was taken from"""
    slice_id: str | None = None
    """Identifier of the slice this recording was made from"""
    fov_id: str | None = None
    """Identifier of the field of view within an individual slice that was
    recorded
    """
    movie_uuid: str | None = None
    """OME UUID of the full tseries"""

    scale_x: float | None = None
    """The size of each pixel in the $x$-axis (in microns)"""
    scale_y: float | None = None
    """The size of each pixel in the $y$-axis (in microns)"""

    ## Specification lenses

    @staticmethod
    @atdata.lens
    def from_generic( s: ts.Frame ) -> 'UncagingFrame':
        
        assert s.metadata is not None
        
        return UncagingFrame(
            # TODO
            uncaged_compound = 'unknown',
            image = s.image,
            t_index = s.metadata['frame']['t_index'],
            t = s.metadata['frame']['t'],
            #
            date_acquired = s.metadata['date_acquired'],
            movie_uuid = s.metadata['uuid'],
        )


#