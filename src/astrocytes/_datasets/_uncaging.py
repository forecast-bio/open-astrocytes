"""Schema and dataset definitions for OpenAstrocytes photochemical uncaging data.

This module defines the typed frame structure for photochemical uncaging experiments,
where neurotransmitters (GABA, glutamate) are released via laser photolysis.
"""

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

UncagingCompound: TypeAlias = Literal[
    'gaba',
    'glu',
    'laser_only',
    'unknown',
]

_COMPOUND_ALIASES: dict[UncagingCompound, list[str]] = {
    'gaba': [
        'gaba',
        'rubigaba',
    ],
    'glu': [
        'glu',
        'glutamate',
        'rubiglu',
        'rubiglutamate',
    ],
    'laser_only': [
        'norubi',
    ]
}
"""Commonly-sed shortcuts and typos for translating to standardized compound values"""

## Sample types

@dataclass
class UncagingFrame( atdata.PackableSample, ExperimentFrame ):
    """Individual imaging frame captured during a photochemical uncaging experiment.

    Represents a single frame from a time-series recording where caged neurotransmitters
    (GABA, glutamate) are released via focused laser photolysis. Includes timing information,
    experimental metadata, and spatial calibration data.
    """
    ##

    uncaged_compound: UncagingCompound
    """The compound uncaged during the experiment for this movie"""
    image: NDArray
    """Image data for the captured frame"""
    t_index: int
    """Frame index in the overall sequence of the original recording"""
    t: float
    """Time (in seconds) this frame was captured after the start of the original recording"""

    t_intervention: float | None = None
    """Time (in seconds) at which the compound was applied; `None` indicates value unknown
    
    TODO: Make required in import scripts
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
    def from_generic( s: Frame ) -> 'UncagingFrame':
        return _specify_uncaging( s )

## Register lenses

def _extract_compound_from_filename( fn: str ) -> UncagingCompound:
    """Extract the uncaged compound type from a filename.

    Performs case-insensitive matching against known compound aliases to identify
    which neurotransmitter (GABA, glutamate) or control condition (laser_only) was used.

    Args:
        fn: Filename or path to parse

    Returns:
        The standardized compound name, or 'unknown' if no match found
    """
    for candidate, aliases in _COMPOUND_ALIASES.items():
        for alias in aliases:
            if alias.lower() in fn.lower():
                return candidate
    return 'unknown'

def _extract_is_test_from_filename( fn: str ) -> bool:
    """Determine if a recording is a test based on filename conventions.

    Uses domain-specific knowledge of file naming patterns to identify test recordings.

    Args:
        fn: Filename or path to parse

    Returns:
        True if the file appears to be from a test recording, False otherwise
    """
    if 'TEST' in fn:
        return True
    return False

@atdata.lens
def _specify_uncaging( s: Frame ) -> UncagingFrame:
    """Convert a generic Frame to a typed UncagingFrame.

    Extracts experiment-specific metadata (uncaged compound, timing, spatial scales, etc.)
    from the generic Frame's metadata dictionary and source filename.

    Args:
        s: A generic toile.Frame with raw imaging data and metadata

    Returns:
        A fully-typed UncagingFrame with extracted experimental metadata

    Raises:
        AssertionError: If required metadata fields are missing from the source frame
    """

    # TODO More elegant validation?
    assert s.metadata is not None, 'Source frame has no metadata'
    assert 'frame' in s.metadata, 'No frame index information available'
    assert (
        't_index' in s.metadata['frame']
        and 't' in s.metadata['frame']
    ), 'Timing information not in frame metadata'

    return UncagingFrame(
        # TODO Correctly parse metadata
        uncaged_compound = _extract_compound_from_filename( s.metadata.get( '_source_filename', '' ) ),
        image = s.image,
        t_index = s.metadata['frame']['t_index'],
        t = s.metadata['frame']['t'],
        #
        # TODO These are based on a priori knowledge of the input datasets; generalize
        # t_intervention = 300. #s
        t_intervention = None,
        is_test = _extract_is_test_from_filename( s.metadata.get( '_source_filename', '' ) ),
        #
        date_acquired = s.metadata.get( 'date_acquired', None ),
        movie_uuid = s.metadata.get( 'uuid', None ),
        #
        scale_x = s.metadata.get( 'scale_x', None ),
        scale_y = s.metadata.get( 'scale_y', None ),
    )