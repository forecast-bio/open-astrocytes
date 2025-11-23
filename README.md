# open-astrocytes

**Open data and models for astrocyte dynamics**

A Python library for discovering, loading, and processing experimental imaging datasets from astrocyte neuroscience research. Built on a cloud-hosted data repository with serverless compute backends for large-scale image analysis.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

## Features

- **Unified Data Discovery**: Access experimental datasets through a single `Hive` interface backed by cloud-hosted manifests
- **Type-Safe Schemas**: Strongly-typed dataclasses for different experiment types (bath application, photochemical uncaging)
- **Lens Transformations**: Composable data pipelines for converting raw frames to typed experiments
- **Vision Transformer Embeddings**: GPU-accelerated DINOv3 embeddings via Modal serverless infrastructure
- **Incremental PCA**: Memory-efficient dimensionality reduction for large-scale patch embeddings
- **WebDataset Format**: Streaming-friendly TAR archives for efficient cloud storage and access

## Installation

```bash
# Install the core package
pip install astrocytes

# Or with uv (recommended for development)
uv pip install astrocytes
```

**Requirements**: Python 3.12 or 3.13

## Quick Start

```python
import astrocytes

# Access the data repository
hive = astrocytes.Hive()

# Load a dataset via shortcuts
dataset = astrocytes.data.bath_application

# Iterate through frames
for frame in dataset.ordered(batch_size=None):
    print(f"Frame at t={frame.t:.1f}s, compound={frame.applied_compound}")
    # frame.image is a numpy array of raw 2P imaging data
```

## Architecture

### Three-Tier Data Abstraction

The library uses a layered approach to organize imaging data:

```
┌─────────────────────────────────────────────────┐
│  Tier 1: Generic (toile.Frame)                 │
│  Raw imaging data with minimal structure       │
└─────────────────┬───────────────────────────────┘
                  │ Lens Transformation
┌─────────────────▼───────────────────────────────┐
│  Tier 2: Typed Experiments                     │
│  BathApplicationFrame, UncagingFrame, etc.     │
│  Domain-specific metadata extracted            │
└─────────────────┬───────────────────────────────┘
                  │ Backend Processing
┌─────────────────▼───────────────────────────────┐
│  Tier 3: Derived Results                       │
│  EmbeddingResult, EmbeddingPCResult            │
│  Vision transformer outputs, PCA projections   │
└─────────────────────────────────────────────────┘
```

### The Hive Pattern

The `Hive` class serves as the main entry point, fetching a YAML manifest from the cloud and organizing datasets hierarchically:

```python
hive = astrocytes.Hive()  # Fetches default manifest from data.forecastbio.cloud

# Navigate the hierarchy
generic_frames = hive.index.generic.bath_application.dataset
embeddings = hive.index.embeddings.bath_application.dataset
pca_reduced = hive.index.patch_pcs.bath_application.dataset
```

## Usage Examples

### Working with Typed Experiments

Convert generic frames to experiment-specific types using lens transformations:

```python
import astrocytes
from astrocytes.schema import BathApplicationFrame

# Load generic frames
generic_dataset = astrocytes.data.bath_application

# Apply lens transformation to get typed frames
typed_dataset = generic_dataset.map(BathApplicationFrame.from_generic)

# Now iterate with full type information
for frame in typed_dataset.ordered(batch_size=None):
    print(f"Compound: {frame.applied_compound}")
    print(f"Time: {frame.t:.2f}s (intervention at {frame.t_intervention}s)")
    print(f"Mouse: {frame.mouse_id}, Slice: {frame.slice_id}")
    print(f"Image shape: {frame.image.shape}")
    print(f"Pixel scale: {frame.scale_x}μm × {frame.scale_y}μm")
```

### Computing Embeddings (Modal Backend)

Process images through the DINOv3 vision transformer:

```python
from astrocytes.backend import embed

# Initialize the Modal app
app = embed.app

# Process a dataset
with app.run():
    embedder = embed.ImageEmbedder()
    output_path = embedder.process.remote(
        wds_url='https://data.example.com/bath-app.tar',
        output_stem='bath-app-embeddings',
        batch_size=32,
        kind='Frame',
        verbose=True
    )
    print(f"Embeddings saved to: {output_path}")
```

### Training Incremental PCA

Reduce embedding dimensionality with streaming PCA:

```python
from astrocytes.backend import pca

# Initialize the Modal app
app = pca.app

# Train a new PCA model
with app.run():
    model_id = pca.ipca.remote(
        wds_url='https://data.example.com/embeddings.tar',
        output_stem='pca-model',
        n_components=64,
        batch_size=20_480,
        n_batches=100,  # Process 100 batches
        verbose=True
    )
    print(f"Model ID: {model_id}")

    # Resume training on the same model
    model_id = pca.ipca.remote(
        wds_url='https://data.example.com/more-embeddings.tar',
        output_stem='pca-model',
        model_id=model_id,  # Continue training
        n_batches=50,
        verbose=True
    )
```

### Projecting Embeddings to PCA Space

Apply a trained PCA model to new embeddings:

```python
import numpy as np
from astrocytes import data
from astrocytes.schema import patch_pc_projector

# Load trained PCA model (from Modal volume or local file)
pca_model = np.load('ipca-model.npz', allow_pickle=True)['ipca'].item()
components = pca_model.components_

# Create projection lens
projector = patch_pc_projector(components)

# Apply to embedding dataset
embeddings = data.bath_application_embeddings
reduced = embeddings.map(projector)

# Iterate through reduced embeddings
for result in reduced.ordered(batch_size=None):
    print(f"Patch PCs shape: {result.patch_pcs.shape}")  # (h, w, n_components)
```

### Experiment Types

#### Bath Application

Experiments where compounds are applied to the bath solution:

```python
from astrocytes.schema import BathApplicationFrame, BathApplicationCompound

# Compounds: 'baclofen', 'tacpd', 'unknown'
for frame in typed_dataset.ordered(batch_size=None):
    if frame.applied_compound == 'baclofen':
        # Analyze GABA_B receptor activation
        pass
```

#### Photochemical Uncaging

Experiments using laser photolysis to release caged neurotransmitters:

```python
from astrocytes.schema import UncagingFrame, UncagingCompound

dataset = astrocytes.data.uncaging
typed = dataset.map(UncagingFrame.from_generic)

# Compounds: 'gaba', 'glu', 'laser_only', 'unknown'
for frame in typed.ordered(batch_size=None):
    if frame.uncaged_compound == 'glu':
        # Analyze glutamate uncaging response
        pass
```

## Dataset Shortcuts

For convenience, common dataset combinations are available directly:

```python
import astrocytes

# Generic datasets (toile.Frame)
astrocytes.data.bath_application
astrocytes.data.uncaging

# Derived datasets (processed)
astrocytes.data.bath_application_embeddings   # EmbeddingResult
astrocytes.data.bath_application_patch_pcs    # EmbeddingPCResult
```

## Development Setup

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-org/open-astrocytes.git
cd open-astrocytes

# Install with development dependencies using uv
uv sync --locked --all-extras --dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=astrocytes --cov-report=html
```

### Working with Modal Backends

The embedding and PCA backends run on [Modal](https://modal.com) serverless infrastructure:

```bash
# Install Modal CLI
pip install modal

# Authenticate with Modal
modal token new

# Set your Modal environment
modal config set-environment <your-environment>

# Deploy the embedding backend
uv run modal deploy src/astrocytes/backend/embed.py

# Deploy the PCA backend
uv run modal deploy src/astrocytes/backend/pca.py
```

### Testing Modal Functions Locally

```bash
# Run embedding backend locally (uses Modal local mode)
uv run modal run src/astrocytes/backend/embed.py

# Run PCA backend locally
uv run modal run src/astrocytes/backend/pca.py
```

## Project Structure

```
open-astrocytes/
├── src/astrocytes/
│   ├── __init__.py              # Main package entry point
│   ├── schema.py                # Public schema API
│   ├── _datasets/               # Dataset management
│   │   ├── __init__.py          # Hive and DatasetIndex
│   │   ├── _common.py           # Base classes
│   │   ├── _bath_application.py # Bath application schema
│   │   ├── _uncaging.py         # Uncaging schema
│   │   ├── _embeddings.py       # Embedding schemas
│   │   └── _future.py           # Future expansions
│   └── backend/                 # Modal compute backends
│       ├── embed.py             # DINOv3 vision transformer
│       └── pca.py               # Incremental PCA
├── tests/                       # Test suite
├── pyproject.toml               # Project metadata
└── README.md                    # This file
```

## Key Dependencies

- **[atdata](https://github.com/forecast-bio/atdata)**: Core dataset abstraction and lens transformations
- **[toile](https://github.com/forecast-bio/toile)**: Generic imaging frame schema
- **[Modal](https://modal.com)**: Serverless compute platform for GPU workloads
- **[transformers](https://huggingface.co/docs/transformers)**: HuggingFace library for default transformer embedding models used here
- **scikit-learn**: Implementation of a few standard data science techniques used here
- **webdataset**: Streaming TAR dataset format

## Data Repository

The default data repository is hosted at:
```
https://data.forecastbio.cloud/open-astrocytes/
```

The manifest is automatically fetched when you create a `Hive()` instance. You can specify a custom repository location to use a separate, cloned instance:

```python
hive = astrocytes.Hive(root='https://my-custom-repo.com/astrocytes')
```

## Contributing

Contributions are welcome! To add a new experiment type:

1. Create a new schema module in `src/astrocytes/_datasets/_your_experiment.py`
2. Define a typed frame class inheriting from `ExperimentFrame`
3. Implement the `from_generic()` lens transformation
4. Add the dataset to `DatasetIndex` in `_datasets/__init__.py`
5. Export types in `schema.py`
6. Add tests in `tests/test_datasets.py`

See [CLAUDE.md](./CLAUDE.md) for detailed development guidelines.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{open_astrocytes,
  title = {open-astrocytes: Open data and models for astrocyte dynamics},
  author = {Levesque, Maxine},
  year = {2025},
  url = {https://github.com/your-org/open-astrocytes}
}
```

## License

[To be determined]

## Acknowledgments

Developed by [Forecast](https://forecast.bio/research).
