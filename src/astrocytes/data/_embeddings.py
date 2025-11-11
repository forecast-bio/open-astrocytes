"""TODO"""

##
# Imports

from dataclasses import dataclass
import atdata

from typing import (
    Any,
)
from numpy.typing import NDArray


##
# Schema

## Sample types
# TODO Add task-specific metadata classes

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