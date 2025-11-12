"""Schema and dataset definitions for OpenAstrocytes bath application data"""

##
# Imports

import atdata

from dataclasses import dataclass

from toile.schema import Frame
from ._common import ExperimentFrame

from typing import (
    TypeAlias,
    Literal,
)
from numpy.typing import NDArray


##
# Schema

## Constants

BathApplicationCompound: TypeAlias = Literal[
    'baclofen',
    'tacpd',
    'unknown',
]

_COMPOUND_ALIASES: dict[BathApplicationCompound, list[str]] = {
    'baclofen': [
        'baclofen',
        'bacloffen',
    ],
    'tacpd': [
        'tacpd',
        'tcapd',
    ],
}
"""Commonly-used shortcuts and typos for translating to standardized compound values"""

## Sample types

# TODO Update to `@atdata.packable` once `atdata` moves to new packable decorator
@dataclass
class BathApplicationFrame( atdata.PackableSample, ExperimentFrame ):
    """Individual imaging frame captured during a bath application experiment"""
    ##

    applied_compound: BathApplicationCompound
    """The compound applied during the experiment for this movie"""
    image: NDArray
    """Image data for the captured frame"""
    t_index: int
    """Frame index in the overall sequence of the original recording"""
    t: float
    """Time (in seconds) this frame was captured after the start of the original recording"""

    t_intervention: float | None = None
    """Time (in seconds) at which the compound was applied; `None` indicates value unknown
    
    TODO: Make required in importlens scripts
    """
    is_test: bool | None = None
    """Whether this frame was acquired during a test
    
    TODO: Make required in import/lens scripts
    """

    date_acquired: str | None = None
    """ISO timestamp at approximately when the experiment was performed"""

    mouse_id: str | None = None
    """Identifier of the mouse this slice was taken from"""
    slice_id: str | None = None
    """Identifier of the slice this recording was made from"""
    fov_id: str | None = None
    """Identifier of the field of view within an individual slice that was recorded"""
    movie_uuid: str | None = None
    """OME UUID of the full tseries"""

    scale_x: float | None = None
    """The size of each pixel in the $x$-axis (in microns)"""
    scale_y: float | None = None
    """The size of each pixel in the $y$-axis (in microns)"""

    ## Specification lenses

    @staticmethod
    def from_generic( s: Frame ) -> 'BathApplicationFrame':
        return _specify_bath_application( s )

## Register lenses

def _extract_compound_from_filename( fn: str ) -> BathApplicationCompound:
    """(Note: case-insensitive matching)"""
    for candidate, aliases in _COMPOUND_ALIASES.items():
        for alias in aliases:
            if alias.lower() in fn.lower():
                return candidate
    return 'unknown'

def _extract_is_test_from_filename( fn: str ) -> bool:
    """TODO Based on a priori knowledge about file structure"""
    if 'TEST' in fn:
        return True
    return False

@atdata.lens
def _specify_bath_application( s: Frame ) -> BathApplicationFrame:
    
    # TODO More elegant validation?
    assert s.metadata is not None, 'Source frame has no metadata'
    assert 'frame' in s.metadata, 'No frame index information available'
    assert (
        't_index' in s.metadata['frame']
        and 't' in s.metadata['frame']
    ), 'Timing information not in frame metadata'

    return BathApplicationFrame(
        # TODO Correctly parse metadata
        applied_compound = _extract_compound_from_filename( s.metadata.get( '_source_filename', '' ) ),
        image = s.image,
        t_index = s.metadata['frame']['t_index'],
        t = s.metadata['frame']['t'],
        #
        # TODO These are based on a priori knowledge of the input datasets; generalize
        t_intervention = 300., #s
        is_test = _extract_is_test_from_filename( s.metadata.get( '_source_filename', '' ) ),
        #
        date_acquired = s.metadata.get( 'date_acquired', None ),
        movie_uuid = s.metadata.get( 'uuid', None ),
        #
        scale_x = s.metadata.get( 'scale_x', None ),
        scale_y = s.metadata.get( 'scale_y', None ),
    )


#