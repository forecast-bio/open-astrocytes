"""OpenAstrocyte-specific dataset schemas.

This module provides the public API for all dataset schema types and transformations
used in the astrocytes package. It exports:

Experiment Frame Types:
    - BathApplicationFrame: Individual frames from bath application experiments
    - BathApplicationCompound: Type alias for bath application compound names
    - UncagingFrame: Individual frames from photochemical uncaging experiments
    - UncagingCompound: Type alias for uncaging compound names

Derived Result Types:
    - EmbeddingResult: Vision transformer embedding outputs (cls + patches)
    - EmbeddingPCResult: PCA-reduced patch embeddings

Transformation Functions:
    - patch_pc_projector: Creates a lens for projecting embeddings to PCA space
"""

##
# Expose types

from ._datasets._bath_application import (
    BathApplicationCompound,
    BathApplicationFrame,
)
from ._datasets._uncaging import (
    UncagingCompound,
    UncagingFrame,
)
from ._datasets._embeddings import (
    EmbeddingResult,
    EmbeddingPCResult,
    #
    patch_pc_projector,
)


#