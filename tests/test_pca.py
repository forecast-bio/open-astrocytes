"""Tests for OpenAstrocytes PCA backend"""

##
# Imports

import pytest

import astrocytes

import modal
import astrocytes.backend.pca as backend


##
# Test units

def test_ipca():
    """Test incremental PCA fitting on Modal backend"""

    assert backend.app.name is not None, \
        'No name specified for PCA Modal backend app'

    ipca = modal.Function.from_name(
        backend.app.name,
        'ipca',
    )

    test_ds = astrocytes.data.bath_application_embeddings
    assert test_ds is not None, \
        'Could not form embedding test dataset'
    
    test_url = test_ds.shard_list[0]
    test_output_stem = '__pytest__ipca'

    model_id = None
    for i in range( 2 ):
        model_id = ipca.remote( test_url, test_output_stem, model_id = model_id,
            #
            batch_size = 256,
            n_batches = 2,
            #
            verbose = True,
        )


#