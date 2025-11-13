"""OpenAstrocytes image embedding backend on Modal"""

##
# Imports

import modal

import dotenv
from pathlib import Path

from dataclasses import dataclass
import atdata

from numpy.typing import NDArray


##
# Constants

MINUTES = 60

# Hyperparameters
# TODO Migrate to `chz` for config

MODAL_APP_NAME = 'astrocytes-backend--embed'

MAX_CONTAINERS = 5

# Only process this number of copies of the model simultaneously, to stay
# within GPU VRAM (for larger embedding models)
MAX_INPUTS_PER_CONTAINER = 1

# how long should we stay up with no requests?
SCALEDOWN_WINDOW = 3 * MINUTES
# how long should we wait for startup + single embed execution?
TIMEOUT = 10 * MINUTES


##
# Modal app setup

app = modal.App( MODAL_APP_NAME )

# Image

_image = (
    modal.Image
        .debian_slim(
            python_version = '3.12',
        )
        .apt_install(
            'git',
            'curl',
        )
        .uv_pip_install(
            'torch',
            'torchvision',
            # Install from source needed for some features
            'git+https://github.com/huggingface/transformers',
            # 'transformers',
            'pillow',
            'numpy',
            'webdataset',
            'pandas',
            'pydantic',
            'numpydantic',
            'accelerate',
            'atdata',
            'toile',
            'python-dotenv',
        )
)

# Volumes

hf_cache_path = '/root/.cache/huggingface'
hf_cache_vol = modal.Volume.from_name( f'{MODAL_APP_NAME}--huggingface-cache',
    create_if_missing = True,
)

output_data_path = '/root/data/output'
output_data_vol = modal.Volume.from_name( f'{MODAL_APP_NAME}--output-data',
    create_if_missing = True,
)

# Function call configuration

_embedding_config = {
    'cpu': 4.,
    # 'gpu': 'A10',
    'gpu': 'A100', # we go bigly
    # TODO figure out how Modal is doing this, ignores this arg rn
    # 'memory': 10 * 1_024,
    
    'enable_memory_snapshot': True,
    'experimental_options': {
        "enable_gpu_snapshot": True,
    },

    'scaledown_window': SCALEDOWN_WINDOW,
    'timeout': TIMEOUT,

    'volumes': {
        hf_cache_path: hf_cache_vol,
        output_data_path: output_data_vol,
    },

    'secrets': [
        # TODO Make more public-friendly nomenclature
        modal.Secret.from_name( 'testing-hf' ),
    ]
}

## Lifecycle management for embedding container

from typing import (
    Optional,
    Dict,
    Any,
    TypeAlias,
    Type,
    Literal,
    Iterable,
)
# from numpy.typing import NDArray

EmbeddableSampleType: TypeAlias = Literal[
    'Frame',
    'BathApplicationFrame',
]

##

@dataclass
class EmbeddingResult( atdata.PackableSample ):
    """TODO"""
    cls_embedding: NDArray
    registers: NDArray | None = None
    patches: NDArray | None = None
    #
    metadata: dict[str, Any] | None = None

##

@app.cls(
    image = _image,
    max_containers = MAX_CONTAINERS,

    **_embedding_config
)
@modal.concurrent(
    max_inputs = MAX_INPUTS_PER_CONTAINER,
)
class ImageEmbedder:
    """TODO"""
    ##

    import atdata

    # TODO Parameterize
    MODEL_NAME = 'facebook/dinov3-vit7b16-pretrain-lvd1689m'
    # MODEL_NAME = 'facebook/dinov3-vits16-pretrain-lvd1689m'
    # MODEL_NAME = 'facebook/dinov2-large'
    
    @modal.enter(
        snap = True,
    )
    def _container_start( self ):

        from transformers import (
            AutoImageProcessor,
            AutoModel,
        )
        import torch

        #

        torch.set_float32_matmul_precision( 'high' )

        #

        # Load model
        self.processor = AutoImageProcessor.from_pretrained( self.MODEL_NAME )
        self.model = AutoModel.from_pretrained( self.MODEL_NAME, device_map = 'auto' )
        # self.model = torch.compile( self.model )

        # Cache model hyperparameters
        self.patch_size = self.model.config.patch_size
        self.hidden_size = self.model.config.hidden_size
        self.num_register_tokens = self.model.config.num_register_tokens

        self.model = torch.compile( self.model )

        ##

        self.output_path = Path( output_data_path ) / 'image-embeddings'
        self.output_path.mkdir( parents = True, exist_ok = True )

    ##

    def _create_dataloader( self, url: str,
            sample_type: Type[atdata.PackableSample],
            batch_size: int | None = 32,
            num_workers: int = 4,
        ):

        import numpy as np
        import torch

        import webdataset as wds
        import atdata
        import toile.schema as ts

        import astrocytes
        import astrocytes.schema as os

        from numpy.typing import NDArray
        
        from PIL import Image

        #

        def _process_image( v: NDArray ):
            """Runs `self.processor` on the 'image' part of an input dataset item"""

            # Form PIL RGB image from grayscale
            # TODO Check dims to donly do this for grayscale images
            rgb_array = np.stack( [v, v, v], axis = -1 )

            # TODO determine scaling in preprocessing
            if rgb_array.dtype != np.uint8:

                if rgb_array.dtype == np.uint16:
                    max_val = np.max( rgb_array )
                    if max_val < 256:
                        rgb_array = rgb_array.astype( np.uint8 )
                    else:
                        tmp = (255. / max_val) * rgb_array.astype( float )
                        tmp = np.floor( tmp )
                        rgb_array = tmp.astype( np.uint8 )
                
                else:
                    max_val = np.max( rgb_array )
                    tmp = rgb_array * (256 / max_val)
                    tmp = np.floor( tmp )
                    rgb_array = tmp.astype( np.uint8 )

                # new_max_val = np.max( rgb_array )
            
            cur_image = Image.fromarray( rgb_array )
            
            return self.processor(
                images = cur_image,
                return_tensors = "pt",
            ).pixel_values


        if sample_type == ts.Frame:

            def _f1( x: ts.Frame ) -> dict:
                ret = dict()

                # TODO This speaks to a larger issue in the way channels are handled in the OA data right now
                if len( x.image.shape ) == 2:
                    ret['image'] = _process_image( x.image )
                elif len( x.image.shape ) == 3:
                    ret['image'] = _process_image( x.image[0] )
                else:
                    raise ValueError( f'Unknown image shape: {x.image.shape}' )
                
                ret['metadata'] = x.metadata
                return ret
            
            _process_sample = _f1
        
        elif sample_type == os.BathApplicationFrame:

            def _f2( x: os.BathApplicationFrame ) -> dict:
                ret = dict()

                # TODO This speaks to a larger issue in the way channels are handled in the OA data right now
                if len( x.image.shape ) == 2:
                    ret['image'] = _process_image( x.image )
                elif len( x.image.shape ) == 3:
                    ret['image'] = _process_image( x.image[0] )
                else:
                    raise ValueError( f'Unknown image shape: {x.image.shape}' )

                # TODO Add reasonable dict reduction built-in to types
                ret['metadata'] = {
                    'applied_compound': x.applied_compound,
                    't_index': x.t_index,
                    't': x.t,
                    'date_acquired': x.date_acquired,
                    'mouse_id': x.mouse_id,
                    'slice_id': x.slice_id,
                    'fov_id': x.fov_id,
                    'movie_uuid': x.movie_uuid,
                    'scale_x': x.scale_x,
                    'scale_y': x.scale_y,
                }
                return ret
            
            _process_sample = _f2
        
        else:
            raise ValueError( f'Unsupported sample type: {sample_type}' )
        
        dataset = atdata.Dataset[sample_type]( url )
        
        pipeline = dataset.ordered( batch_size = None )
        pipeline.append( wds.filters.map( _process_sample ) )
        # Form batch after applying sample-wise preprocessing filter
        pipeline.append( wds.filters.batched( batch_size ) )
        return pipeline
        
        # TODO get 
        # return wds.compat.WebLoader( pipeline, num_workers = num_workers )

    ##

    @modal.method()
    def process( self, wds_url: str, output_stem: str,
            batch_size: int = 32,
            n_batches: Optional[int] = None,
            #
            kind: EmbeddableSampleType = 'Frame',
            #
            sharded: bool = False,
            verbose: bool = False,
        ) -> str:
        """TODO"""

        import torch

        from functools import reduce

        import atdata
        import toile.schema as ts
        import astrocytes.schema as os

        import webdataset as wds

        #

        def _vprint( *args, **kwargs ):
            if verbose:
                print( *args, **kwargs )

        sample_type: Type[atdata.PackableSample]
        if kind == 'Frame':
            sample_type = ts.Frame
        elif kind == 'BathApplicationFrame':
            sample_type = os.BathApplicationFrame
        else:
            raise ValueError( f'Unrecognized sample type option: {kind}' )

        #

        _vprint( 'Creating dataloader' )

        # input_url = (Path( self.input_bucket ) / name).as_posix()
        dataloader = self._create_dataloader( wds_url,
            sample_type = sample_type,
            batch_size = batch_size,
        )

        ret_batches: list[list[EmbeddingResult]] = []
        for i_batch, cur_batch in enumerate( dataloader ):

            # print( cur_batch )

            _vprint( f'Starting batch {i_batch}' )
            cur_images = cur_batch['image']
            cur_metadatas = cur_batch['metadata']
            _vprint( '    Loaded' )

            # Check model preprocessing
            inputs = torch.squeeze( cur_images ).to( 'cuda' )
            _vprint( '        Shape:', inputs.shape )

            batch_size, _num_channels, img_height, img_width = inputs.shape
            num_patches_height = img_height // self.patch_size
            num_patches_width = img_width // self.patch_size
            num_patches_flat = num_patches_height * num_patches_width

            _vprint( '    Running model' )
            with torch.inference_mode():
                outputs = self.model( pixel_values = inputs )
            _vprint( '        Done' )

            _vprint( '    Forming outputs' )
            last_hidden_states = outputs.last_hidden_state.to( 'cpu' )
            assert (
                last_hidden_states.shape == (
                    batch_size,
                    1 + self.num_register_tokens + num_patches_flat,
                    self.hidden_size
                )
            ), 'Improper output hidden state shape'

            # Split outputs
            cls_token = last_hidden_states[:, 0, :]
            
            register_features_flat = last_hidden_states[:, 1:(1 + self.num_register_tokens), :]
            register_features = register_features_flat.unflatten( 1, (self.num_register_tokens,) )

            patch_features_flat = last_hidden_states[:, 1 + self.num_register_tokens:, :]
            patch_features = patch_features_flat.unflatten( 1, (num_patches_height, num_patches_width) )

            cur_batch_ret = [
                EmbeddingResult(
                    cls_embedding = cls_token[i, :].numpy(),
                    registers = register_features[i, :, :].numpy(),
                    patches = patch_features[i, :, :].numpy(),
                    #
                    metadata = cur_metadatas[i],
                )
                for i in range( cls_token.shape[0] )
            ]
            ret_batches.append( cur_batch_ret )

            _vprint( '        Done' )
            _vprint( '    Batch Done' )

            if n_batches is not None:
                if i_batch >= n_batches:
                    _vprint( 'Reached batch limit' )
                    break
        
        _vprint( '    Exporting outputs' )
        output_dir = self.output_path / output_stem
        output_dir.mkdir( parents = True, exist_ok = True )

        if sharded:

            output_shard_pattern = f'{output_stem}-%06d.tar'
            output_pattern = (output_dir / output_shard_pattern).as_posix()
            
            with wds.writer.ShardWriter( output_pattern ) as dest:
                for cur_batch in ret_batches:
                    for cur_output in cur_batch:
                        dest.write( cur_output.as_wds )
            
            output_loc = (
                output_dir
                / (f'{output_stem}-' + '{shard_id}.tar')
            ).as_posix()
        
        else:

            output_filename = f'{output_stem}.tar'
            output_path = (output_dir / output_filename).as_posix()

            with wds.writer.TarWriter( output_path ) as dest:
                for cur_batch in ret_batches:
                    for cur_output in cur_batch:
                        dest.write( cur_output.as_wds )
            
            output_loc = output_path
        
        #

        _vprint( '        Done')
        _vprint( 'All Done! Output written to:' )
        _vprint( f'    {output_loc}' )
        
        return output_loc

    ##

    @modal.exit()
    def _container_shutdown( self ):
        pass


#