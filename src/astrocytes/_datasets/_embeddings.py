"""TODO"""

##
# Imports

from dataclasses import dataclass
import atdata

from ._common import (
    DatasetInfo,
    ST,
)

from typing import (
    Any,
    Type,
)
from numpy.typing import NDArray


##
# Index

@dataclass
class EmbeddingsDatasetIndex:
    """TODO"""
    ##
    def __init__( self,
                config: dict[str, Any],
                hive_root: str = '',
            ):
        """TODO"""

        # TODO Shortcut; implies better way to generalize
        def _typed_info( name: str, sample_type: Type[ST] = EmbeddingResult ) -> DatasetInfo[ST] | None:
            return DatasetInfo._parse(
                config.get( name ), 'embeddings/' + name,
                sample_type = sample_type,
                hive_root = hive_root,
            )
        
        self.bath_application = _typed_info( 'bath_application' )
        self.uncaging = _typed_info( 'uncaging' )

@dataclass
class PatchPCsDatasetIndex:
    """TODO"""
    ##
    def __init__( self,
                config: dict[str, Any],
                hive_root: str = '',
            ):
        """TODO"""

        # TODO Shortcut; implies better way to generalize
        def _typed_info( name: str, sample_type: Type[ST] = EmbeddingPCResult ) -> DatasetInfo[ST] | None:
            return DatasetInfo._parse(
                config.get( name ), 'patch-pcs/' + name,
                sample_type = sample_type,
                hive_root = hive_root,
            )
        
        self.bath_application = _typed_info( 'bath_application' )
        self.uncaging = _typed_info( 'uncaging' )


##
# Schema

## Sample types
# TODO Add task-specific metadata breakout classes

@dataclass
class EmbeddingResult( atdata.PackableSample ):
    """TODO"""
    ##
    cls_embedding: NDArray
    """TODO"""
    #
    registers: NDArray | None = None
    """TODO"""
    patches: NDArray | None = None
    """TODO"""
    #
    metadata: dict[str, Any] | None = None
    """TODO"""

@dataclass
class EmbeddingPCResult( atdata.PackableSample ):
    """TODO"""
    ##
    patch_pcs: NDArray
    """TODO"""
    #
    metadata: dict[str, Any] | None = None
    """TODO"""

# Lenses

def patch_pc_projector( components: NDArray ) -> atdata.Lens:
    """Registers and returns a lens that computes patch PCs for embeddings"""

    @atdata.lens
    def _embedding_patch_pcs( source: EmbeddingResult ) -> EmbeddingPCResult:
        assert source.patches is not None, \
            'Source embedding result has no patch embeddings'
        
        return EmbeddingPCResult(
            patch_pcs = (components @ source.patches.T).T,
            metadata = source.metadata,
        )

    return _embedding_patch_pcs


#