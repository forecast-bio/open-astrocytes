"""OpenAstrocytes embedding PCA backend on Modal"""

##
# Imports

import modal


##
# Constants

MINUTES = 60

# Hyperparameters
# TODO Migrate to `chz` for config

MODAL_APP_NAME = 'astrocytes-backend--pca'

# how long should we wait for startup + single execution?
TIMEOUT = 5 * MINUTES


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
            'atdata==0.1.3b2',
            'toile',
            'python-dotenv',
            'scikit-learn',
            'fastparquet',
            'requests',
        )
        .add_local_python_source(
            'openastros',
            copy = True
        )
)

# Volumes

output_data_path = '/root/data/output'
output_data_vol = modal.Volume.from_name(
    f'{MODAL_APP_NAME}--output-data--pca',
    create_if_missing = True,
)

remote_secret = modal.Secret.from_name( 'public-data--r2' )
remote_bucket_path = '/root/data/remote/forecast'
remote_bucket_vol = modal.CloudBucketMount(
    bucket_name = 'public-data',
    # TODO
    bucket_endpoint_url = 'https://f5bf77c06cb35b5136ff6d61ab4b7dbc.r2.cloudflarestorage.com',
    secret = remote_secret,
)


##
# Functions

# TODO

@app.function(
    timeout = TIMEOUT,
    image = image,
    volumes = {
        output_data_path: output_data_vol,
        remote_bucket_path: remote_bucket_vol,
    }
)
def ipca( to: str, chunk_size: int = 100 ) -> bool:
    """TODO"""

    from tqdm import tqdm

    import pandas as pd
    import numpy as np

    from sklearn.decomposition import IncrementalPCA

    import astrocytes

    patch_block_raw = []
    metadata_raw = []
    i_sample = 0

    bath_application = (
        openastros.hive.index
            .embeddings.bath_application
    )
    assert bath_application is not None, \
        'OA hive manifest not properly specified'
    
    dataset = bath_application.dataset

    print( '** Loading data...' )

    for sample in tqdm( dataset.shuffled( batch_size = None ) ):

        try:
            assert sample.metadata is not None
            assert sample.patches is not None

            patch_block_raw.append(
                np.expand_dims( sample.patches, axis = 0 )
            )
            metadata_raw.append( _extract_metadata( sample ) )

            i_sample += 1
            # if i_sample % 10 == 0:
            #     print( '.', end = '' )
        
        except Exception as e:
            print( f'({i_sample}) Skipping incopatible frame - no metadata available' )
            print( e )
        
        ##
        
        if i_sample >= chunk_size:
            break
    
    print( '** Done loading data.' )

    print( 'Collating outputs ...' )

    print( '    numpy ...', end = '' )
    arr_patch_block = np.concat( patch_block_raw )
    print( ' done.' )
    print( '    metadata ...', end = '' )
    df_metadata = pd.DataFrame( metadata_raw )
    print( ' done.' )

    print( '** Saving results' )

    print( '    numpy ...', end = '' )
    output_path = to + '.npz'
    with open( output_path, 'wb' ) as f:
        np.savez_compressed( f, arr_patch_block )
    print( ' done.' )

    print( '    metadata ...', end = '' )
    output_path = to + '.parquet'
    with open( output_path, 'wb' ) as f:
        df_metadata.to_parquet( f )
    print( ' done.' )

    print( '** Done with chunk!' )

    return True


#