"""Tests for OpenAstrocyte datasets"""

##
# Imports

import pytest

import astrocytes

from toile.schema import (
    Frame,
)
from astrocytes.schema import (
    BathApplicationFrame,
    UncagingFrame,
)



##
# Test units

def test_placeholder():
    """TODO A placeholder test to get the CI workflow up"""
    assert True == True

def test_bath_application():
    """TODO"""
    
    dataset = astrocytes.data.bath_application
    assert dataset is not None, \
        'Unable to form uncaging dataset shortcut'
    
    sample = next( x for x in dataset.ordered( batch_size = None ) )
    assert isinstance( sample, Frame ), \
        f'Incorrect type for uncaging dataset generic sample: {type( sample )}'

    dataset_typed = dataset.as_type( BathApplicationFrame )
    sample_typed = next( x for x in dataset_typed.ordered( batch_size = None ) )
    assert isinstance( sample_typed, BathApplicationFrame ), \
        f'Incorrect type for uncaging dataset typed sample: {type( sample_typed )}'

def test_uncaging():
    """TODO"""
    
    dataset = astrocytes.data.uncaging
    assert dataset is not None, \
        'Unable to form uncaging dataset shortcut'
    
    sample = next( x for x in dataset.ordered( batch_size = None ) )
    assert isinstance( sample, Frame ), \
        f'Incorrect type for uncaging dataset generic sample: {type( sample )}'

    dataset_typed = dataset.as_type( UncagingFrame )
    sample_typed = next( x for x in dataset_typed.ordered( batch_size = None ) )
    assert isinstance( sample_typed, UncagingFrame ), \
        f'Incorrect type for uncaging dataset typed sample: {type( sample_typed )}'


#