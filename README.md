# open-astrocytes

**Open data and models for astrocyte dynamics**

A Python library for discovering, loading, and processing experimental imaging datasets from astrocyte neuroscience research. Built on a cloud-hosted data repository.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

## Features

- **Unified Data Discovery**: Access experimental datasets through a single `Hive` interface backed by cloud-hosted manifests
- **Type-Safe Schemas**: Strongly-typed dataclasses for different experiment types (bath application, photochemical uncaging)
- **Lens Transformations**: Composable data pipelines for converting raw frames to typed experiments
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

### Three-Tier Data Organization

The library organizes imaging data in three tiers:

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
                  │
┌─────────────────▼───────────────────────────────┐
│  Tier 3: Derived Results (Pre-computed)        │
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
embeddings = hive.index.embeddings.bath_application.dataset  # Pre-computed embeddings
pca_reduced = hive.index.patch_pcs.bath_application.dataset  # Pre-computed PCA projections
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
typed_dataset = generic_dataset.as_type(BathApplicationFrame)

# Now iterate with full type information
for frame in typed_dataset.ordered(batch_size=None):
    print(f"Compound: {frame.applied_compound}")
    print(f"Time: {frame.t:.2f}s (intervention at {frame.t_intervention}s)")
    print(f"Mouse: {frame.mouse_id}, Slice: {frame.slice_id}")
    print(f"Image shape: {frame.image.shape}")
    print(f"Pixel scale: {frame.scale_x}μm × {frame.scale_y}μm")
```

### Working with Pre-computed Embeddings

The data repository includes pre-computed vision transformer embeddings and PCA projections. You can access these directly or apply custom transformations:

```python
from astrocytes import data

# Access pre-computed embeddings
embeddings = data.bath_application_embeddings
for result in embeddings.ordered(batch_size=None):
    print(f"CLS embedding shape: {result.cls_embedding.shape}")
    print(f"Patch embeddings shape: {result.patches.shape}")  # (h, w, embedding_dim)
    break

# Access pre-computed PCA projections
pca_results = data.bath_application_patch_pcs
for result in pca_results.ordered(batch_size=None):
    print(f"Patch PCs shape: {result.patch_pcs.shape}")  # (h, w, n_components)
    break
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
    # ...
```

#### Photochemical Uncaging

Experiments using two-photon photo-uncaging to release caged neurotransmitters:

```python
from astrocytes.schema import UncagingFrame

dataset = astrocytes.data.uncaging
typed = dataset.map(UncagingFrame.from_generic)

# Compounds: 'gaba', 'glu', 'laser_only', 'unknown'
for frame in typed.ordered(batch_size=None):
    if frame.uncaged_compound == 'glu':
        # Analyze glutamate uncaging response
        pass
    # ...
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

```bash
# Clone the repository
git clone https://github.com/forecast-bio/open-astrocytes.git
cd open-astrocytes

# Install with development dependencies using uv
uv sync --locked --all-extras --dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=astrocytes --cov-report=html
```

## Project Structure

```
open-astrocytes/
├── src/astrocytes/
│   ├── __init__.py              # Main package entry point
│   ├── schema.py                # Public schema API
│   └── _datasets/               # Dataset management
│       ├── __init__.py          # Hive and DatasetIndex
│       ├── _common.py           # Base classes
│       ├── _bath_application.py # Bath application schema
│       ├── _uncaging.py         # Uncaging schema
│       ├── _embeddings.py       # Embedding schemas
│       └── _future.py           # Future expansions
├── tests/                       # Test suite
├── pyproject.toml               # Project metadata
└── README.md                    # This file
```

## Key Dependencies

- **[atdata](https://github.com/forecast-bio/atdata)**: Core dataset abstraction and lens transformations
- **[toile](https://github.com/forecast-bio/toile)**: Generic imaging frame schema
- **matplotlib**: Plotting and visualization
- **scikit-image**: Image processing utilities
- **scipy**: Scientific computing tools

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

This project is licensed under the [Mozilla Public License 2.0](LICENSE.md) - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

Developed by [Forecast](https://forecast.bio/research).
