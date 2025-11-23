"""Embedding dataset schemas and transformations.

This module defines:
    - EmbeddingsDatasetIndex and PatchPCsDatasetIndex: Indices for accessing embedding datasets
    - EmbeddingResult: Vision transformer outputs (cls token + register tokens + patch embeddings)
    - EmbeddingPCResult: PCA-reduced patch embeddings
    - PatchEmbeddingTrace: Time-series of patch embedding values
    - patch_pc_projector: Lens factory for projecting embeddings to PCA space
"""

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
    """Index of available embedding result datasets.

    Provides access to vision transformer embedding outputs for different experiment types.
    Each attribute is either a DatasetInfo[EmbeddingResult] or None if unavailable.

    Attributes:
        bath_application: Embeddings from bath application experiments
        uncaging: Embeddings from photochemical uncaging experiments
    """
    ##
    def __init__( self,
                config: dict[str, Any],
                hive_root: str = '',
            ):
        """Initialize the embeddings dataset index.

        Args:
            config: Configuration dictionary mapping experiment types to dataset configs
            hive_root: Base URL for the data repository
        """

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
    """Index of available PCA-reduced patch embedding datasets.

    Provides access to dimensionality-reduced patch embeddings for different experiment types.
    Each attribute is either a DatasetInfo[EmbeddingPCResult] or None if unavailable.

    Attributes:
        bath_application: PCA-reduced patch embeddings from bath application experiments
        uncaging: PCA-reduced patch embeddings from photochemical uncaging experiments
    """
    ##
    def __init__( self,
                config: dict[str, Any],
                hive_root: str = '',
            ):
        """Initialize the patch PCA dataset index.

        Args:
            config: Configuration dictionary mapping experiment types to dataset configs
            hive_root: Base URL for the data repository
        """

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
    """Vision transformer embedding outputs for a single image.

    Contains the complete embedding representation from a vision transformer model
    (e.g., DINOv3), including the CLS token, register tokens, and per-patch embeddings.

    Attributes:
        cls_embedding: Global image embedding from the CLS token (shape: [hidden_size])
        registers: Register token embeddings (shape: [num_registers, hidden_size])
        patches: Per-patch embeddings (shape: [height_patches, width_patches, hidden_size])
        metadata: Experimental metadata carried forward from the source frame
    """
    ##
    cls_embedding: NDArray
    """Global image embedding from the CLS token"""
    #
    registers: NDArray | None = None
    """Register token embeddings"""
    patches: NDArray | None = None
    """Per-patch spatial embeddings"""
    #
    metadata: dict[str, Any] | None = None
    """Experimental metadata from source frame"""

@dataclass
class EmbeddingPCResult( atdata.PackableSample ):
    """PCA-reduced patch embeddings for a single image.

    Contains patch embeddings projected into a lower-dimensional PCA space,
    reducing memory footprint while preserving key variance.

    Attributes:
        patch_pcs: PCA-projected patch embeddings (shape: [height_patches, width_patches, n_components])
        metadata: Experimental metadata carried forward from the source frame
    """
    ##
    patch_pcs: NDArray
    """PCA-projected patch embeddings"""
    #
    metadata: dict[str, Any] | None = None
    """Experimental metadata from source frame"""

#

@dataclass
class PatchEmbeddingTrace( atdata.PackableSample ):
    """Time-series of embedding values for a single image patch.

    Tracks how the embedding representation of a specific spatial patch evolves
    over time across multiple frames of a recording.

    Attributes:
        values: Embedding values over time (shape: [n_timepoints, embedding_dim])
        ts: Timestamps for each value (in seconds)
        i_patch: Vertical (row) index of the patch in the image grid
        j_patch: Horizontal (column) index of the patch in the image grid
        metadata: Experimental metadata
    """
    ##
    values: NDArray
    """Embedding values over time"""
    ts: NDArray
    """Timestamps corresponding to each value"""
    #
    i_patch: int | None = None
    """Vertical index of the patch location in the image grid"""
    j_patch: int | None = None
    """Horizontal index of the patch location in the image grid"""
    #
    metadata: dict[str, Any] | None = None
    """Experimental metadata"""


##
# Lenses

def patch_pc_projector( components: NDArray ) -> atdata.Lens:
    """Create a lens that projects patch embeddings to PCA space.

    Factory function that creates a data transformation lens for reducing the
    dimensionality of patch embeddings using pre-computed PCA components.

    Args:
        components: PCA projection matrix (shape: [n_components, embedding_dim])

    Returns:
        An atdata.Lens that transforms EmbeddingResult → EmbeddingPCResult

    Example:
        >>> # Load PCA components from trained model
        >>> pca_components = load_pca_model()
        >>> # Create projection lens
        >>> projector = patch_pc_projector(pca_components)
        >>> # Apply to dataset
        >>> reduced = embedding_dataset.map(projector)
    """

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