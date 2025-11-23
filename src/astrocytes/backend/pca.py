"""OpenAstrocytes embedding PCA backend on Modal"""

##
# Imports

import modal

from pathlib import Path

from astrocytes.schema import (
    EmbeddingResult,
)


##
# Constants

MINUTES = 60

# Hyperparameters
# TODO Migrate to `chz` for config

MODAL_APP_NAME = 'astrocytes-backend--pca'

# how long should we stay up with no requests?
SCALEDOWN_WINDOW = 1 * MINUTES
# how long should we wait for startup + single execution?
TIMEOUT = 10 * MINUTES


##
# Modal app setup

app = modal.App( MODAL_APP_NAME )

# Image

image = (
    modal.Image
        .debian_slim(
            python_version = '3.12',
        )
        .apt_install(
            'curl',
        )
        .uv_pip_install(
            'tqdm',
            'numpy',
            'pandas',
            'webdataset',
            'python-dotenv',
            'scikit-learn',
            'fastparquet',
            'requests',
            #
            'atdata',
            'toile',
            'astrocytes'
        )
        .env(
            {
                'MKL_NUM_THREADS': '4',
                'OPENBLAS_NUM_THREADS': '4',
                'BLIS_NUM_THREADS': '4',
            }
        )
)

# Volumes

output_data_path = '/root/data/output'
output_data_vol = modal.Volume.from_name(
    f'{MODAL_APP_NAME}--output-data',
    create_if_missing = True,
)


##
# Helpers

# These help narrow down relevant metadata to hold on to for PCA results

_GLOBAL_EXTRACT_KEYS = [
    'sensor',
    'age',
    'sex',
    'slice_id',
    'concentration',
    'compound',
    'recording_id',
]
_SAMPLE_EXTRACT_KEYS = [
    't_index',
    't',
]

def _extract_metadata( x: EmbeddingResult ) -> dict:
    """Extract relevant metadata fields from an EmbeddingResult.

    Filters metadata to retain only experiment-relevant fields, reducing
    storage overhead in PCA result objects.

    Args:
        x: An EmbeddingResult with metadata to extract

    Returns:
        A dictionary with selected global and frame-specific metadata fields
    """
    assert x.metadata is not None
    return dict(
        **{
            k: x.metadata.get( k, None )
            for k in _GLOBAL_EXTRACT_KEYS
        },
        **{
            k: x.metadata.get( 'frame', dict() ).get( k, None )
            for k in _SAMPLE_EXTRACT_KEYS
        }
    )


##
# Functions

# TODO

IPCA_OUTPUT_DIR = (
    Path( output_data_path )
    / 'models' / 'ipca'
).as_posix()
"""Directory within the Modal output drive where IPCA models are stored.

All trained IPCA models are saved to this location in the persistent Modal volume.
"""

IPCA_FILENAME_TEMPLATE = 'ipca-{model_id}.npz'
"""Filename template for IPCA model outputs.

Models are saved as compressed numpy archives with the model_id (with hyphens
replaced by underscores) embedded in the filename.
"""

@app.function(
    timeout = TIMEOUT,
    scaledown_window = SCALEDOWN_WINDOW,
    #
    image = image,
    volumes = {
        output_data_path: output_data_vol,
    },
    #
    cpu = 12.,
    memory = 8_000,
    #
    enable_memory_snapshot = True,
)
def ipca( wds_url: str, output_stem: str,
            model_id: str | None = None,
            #
            n_components: int = 64,
            batch_size: int = 5 * 4_096,
            n_batches: int | None = None,
            #
            verbose: bool = False,
        ) -> str:
    """Train an Incremental PCA model on patch embeddings.

    Streams embedding data from a WebDataset TAR, flattens patch embeddings from
    2D grids into vectors, and performs incremental PCA training in batches.
    Supports resuming training from a previously-saved model checkpoint.

    The function processes embeddings in streaming fashion, making it suitable
    for datasets too large to fit in memory. Each batch of patches is fed to
    scikit-learn's IncrementalPCA, which updates the model incrementally.

    Args:
        wds_url: Source WebDataset URL containing EmbeddingResult samples
        output_stem: Base name for output (currently unused, may be for future metadata)
        model_id: Optional model ID to resume training. If None, starts a new model
            with a randomly-generated UUID.
        n_components: Number of principal components to compute
        batch_size: Number of patch embeddings to accumulate before each PCA update.
            Default of 20,480 patches balances memory usage and convergence.
        n_batches: Maximum number of batches to process (None = process entire dataset)
        verbose: If True, print progress messages during training

    Returns:
        The model_id (UUID string) identifying the saved model. Use this ID to
        resume training or load the model for projections.

    Example:
        >>> # Train a new model
        >>> model_id = ipca.remote(
        ...     'https://data.example.com/embeddings.tar',
        ...     'my-pca',
        ...     n_components=64,
        ...     n_batches=100
        ... )
        >>> # Resume training on the same model
        >>> model_id = ipca.remote(
        ...     'https://data.example.com/more-embeddings.tar',
        ...     'my-pca',
        ...     model_id=model_id,
        ...     n_batches=50
        ... )
    """

    from uuid import uuid4
    from time import sleep

    # from tqdm import tqdm

    import pandas as pd
    import numpy as np
    from sklearn.decomposition import IncrementalPCA

    import atdata

    from astrocytes.schema import (
        EmbeddingResult,
    )


    ##

    def _vprint( *args, **kwargs ):
        if verbose:
            print( *args, **kwargs )


    ##
    # Normalize input args

    if model_id is None:
        # Starting a new run

        model_id = str( uuid4() )

        reducer = IncrementalPCA(
            n_components = n_components,
        )

    else:
        # Restore last run for this model id

        input_path = (
            Path( IPCA_OUTPUT_DIR )
            / IPCA_FILENAME_TEMPLATE.format(
                    model_id = model_id.replace( '-', '_' )
            )
        )
        reducer = np.load( input_path, allow_pickle = True )['ipca'].item()
    

    ##

    dataset = atdata.Dataset[EmbeddingResult]( wds_url )

    #

    # metadata_raw = []

    i_batch_prev = -1
    i_batch = 0

    batch_flat_acc = []
    i_sample_batch = 0

    dot_counter = 0

    for sample in dataset.shuffled( batch_size = None ):

        if i_batch > i_batch_prev:
            _vprint( f'** Starting batch {i_batch + 1}' )
            _vprint( '    Loading data' )
            _vprint( '    ', end = '' )
        i_batch_prev = i_batch

        try:

            # Check the current sample for adding to the IPCA fitting batch
            assert sample.metadata is not None
            assert sample.patches is not None

            cur_flat = np.reshape( sample.patches,
                (
                    sample.patches.shape[0] * sample.patches.shape[1],
                    -1
                )
            )

            batch_flat_acc.append( cur_flat )
            # metadata_raw.append( _extract_metadata( sample ) )
            
            i_sample_batch += cur_flat.shape[0]
            dot_counter += 1
            
            if dot_counter % 10 == 0:
                _vprint( '.', end = '' )
        
        except Exception as e:
            print( f'({i_sample_batch}) Skipping incopatible frame - no metadata available' )
            print( e )
        
        ##
        
        if i_sample_batch >= batch_size:
            _vprint( 'done' )

            # Do an IPCA step with the current data
            _vprint( '    Collating data...', end = '' )
            batch_input_data = np.concat( batch_flat_acc, axis = 0 )
            _vprint( 'done' )
            _vprint( '    Running IPCA partial fit...', end = '' )
            reducer.partial_fit( batch_input_data )
            _vprint( 'done' )

            # Reset for the next batch
            i_batch += 1

            batch_flat_acc = []
            i_sample_batch = 0

            if n_batches is not None:
                if i_batch >= n_batches:
                    # We've completed the requisite number of batches
                    break

    # Save output
    _vprint( '** Saving model ...', end = '' )

    output_path = (
        Path( IPCA_OUTPUT_DIR )
        / IPCA_FILENAME_TEMPLATE.format(
                model_id = model_id.replace( '-', '_' )
        )
    )

    output_path.parent.mkdir( parents = True, exist_ok = True )

    with open( output_path, 'wb' ) as f:
        np.savez_compressed( f,
            # TODO Figure out how to get the linter less angsty
            ipca = reducer,
            #
            allow_pickle = True
        )
    output_data_vol.commit()
    _vprint( 'done' )
    
    return model_id


#