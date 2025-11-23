# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**open-astrocytes** is a Python library for managing open astrocyte neuroscience research data. It provides a unified interface for discovering, loading, and processing experimental imaging datasets through cloud-hosted data repositories and serverless compute backends.

## Common Commands

### Development Setup
```bash
# Install dependencies (uses uv package manager)
uv sync --locked --all-extras --dev

# Install without dev dependencies
uv sync --locked --all-extras
```

### Testing
```bash
# Run all tests with coverage
uv run pytest

# Run tests without coverage report
uv run pytest --no-cov

# Run specific test file
uv run pytest tests/test_datasets.py

# Run specific test function
uv run pytest tests/test_datasets.py::test_bath_application
```

### Modal Backend (Cloud Compute)
```bash
# Configure Modal environment
uv run modal config set-environment <environment-name>

# Deploy/update Modal functions (from src/astrocytes/backend/)
uv run modal deploy embed.py
uv run modal deploy pca.py

# Run Modal functions locally for testing
uv run modal run embed.py
```

### Building and Publishing
```bash
# Build package
uv build

# The CI/CD pipeline automatically publishes to PyPI on GitHub releases
# See .github/workflows/uv-publish-pypi.yml
```

## Architecture

### Three-Tier Dataset Abstraction

The library uses a layered approach to data transformation:

1. **Generic (toile.Frame)**: Raw imaging data with minimal structure
2. **Typed (BathApplicationFrame, UncagingFrame)**: Domain-specific frames with semantic annotations extracted from metadata
3. **Derived (EmbeddingResult, EmbeddingPCResult)**: Outputs from compute backends (embeddings, PCA projections)

### The Hive Pattern

The `Hive()` class is the main entry point:
- Fetches manifest from `https://data.forecastbio.cloud/open-astrocytes/manifest.yml`
- Creates a `DatasetIndex` containing generic, embeddings, and patch_pcs datasets
- Provides `DatasetShortcuts` for convenient access (e.g., `Hive().data.bath_application`)

### Lens-Based Transformations

Data transformations use the `atdata.lens` pattern for composable pipelines:

```python
@atdata.lens
def _specify_bath_application(s: Frame) -> BathApplicationFrame:
    # Converts generic Frame → BathApplicationFrame
```

This allows type-safe, declarative data processing. Lenses are applied via `Dataset.map()` or similar methods.

### Backend Services (Modal)

Two serverless compute functions run on Modal cloud infrastructure:

**ImageEmbedder** (`backend/embed.py`):
- Processes images with DINOv3 vision transformer (`facebook/dinov3-vit7b16-pretrain-lvd1689m`)
- Configured for A100 GPUs with max 5 concurrent containers
- Outputs cls token, register tokens, and per-patch embeddings
- Results stored as WebDataset TAR files

**IPCA** (`backend/pca.py`):
- Incremental PCA for reducing patch embedding dimensions
- Processes streaming data in batches for memory efficiency
- Supports checkpointing via model IDs
- Saves models as compressed numpy files

## Key File Locations

### Source Code Structure
- `src/astrocytes/__init__.py` - Package entry point, exports `Hive` and data shortcuts
- `src/astrocytes/schema.py` - Re-exports all schema types for public API
- `src/astrocytes/_datasets/` - Dataset management and schema definitions:
  - `__init__.py` - Main `DatasetIndex` and `Hive` classes
  - `_common.py` - Generic base classes (`DatasetInfo`, `GenericDatasetIndex`)
  - `_bath_application.py` - Bath application experiment schema
  - `_uncaging.py` - Photochemical uncaging experiment schema
  - `_embeddings.py` - Embedding and projection schemas
- `src/astrocytes/backend/` - Modal-based compute backends:
  - `embed.py` - Image embedding using vision transformers
  - `pca.py` - Incremental PCA for dimensionality reduction

### Configuration
- `pyproject.toml` - Package metadata, dependencies, pytest/coverage config
- `uv.lock` - Locked dependency versions for reproducible builds
- `.github/workflows/uv-test.yaml` - CI test runner with Modal integration
- `.github/workflows/uv-publish-pypi.yml` - PyPI publishing on releases

## Important Conventions

### Python Version Requirements
- Requires Python >=3.12, <3.14
- Uses modern type hints and generics extensively

### Metadata Extraction
- Filenames are parsed using compound alias dictionaries (e.g., 'baclofen', 'bacloffen' → 'baclofen')
- Metadata extracted from nested dictionaries with fallback to None
- Validation currently uses assertions (not custom exceptions)

### Data Format
- All data serialized as WebDataset TAR archives for efficient streaming from cloud storage
- Supports sharded outputs for parallel processing

### Testing Strategy
- Backend code (`src/astrocytes/backend/*`) is excluded from coverage reports (runs on Modal, not locally)
- Modal tests require `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and `MODAL_ENVIRONMENT` environment variables
- CI runs on Ubuntu for all pushes to main/release/* branches and PRs

## Adding a New Experiment Type

1. Create `src/astrocytes/_datasets/_new_experiment.py` with `NewExperimentFrame` class
2. Define compound type and extraction helper functions
3. Implement `_specify_new_experiment()` lens function decorated with `@atdata.lens`
4. Add dataset index to `DatasetIndex.__init__()` in `_datasets/__init__.py`
5. Export schema in `src/astrocytes/schema.py`
6. Add test in `tests/test_datasets.py`

## Key Dependencies

- **atdata**: Core dataset library providing `Dataset[T]`, `@lens`, and `PackableSample` abstractions
- **toile**: Defines generic `Frame` schema (base for all imaging data)
- **modal**: Serverless function platform for GPU-accelerated compute
- **transformers**: HuggingFace library for DINOv3 vision model
- **torch**: Deep learning framework required by transformers
- **scikit-learn**: Provides IncrementalPCA implementation
- **webdataset**: Streaming TARball dataset format reader/writer

## Notes

- Coverage reports are generated at `htmlcov/` after running tests
- Backend functions use hardcoded constants for Modal configuration (GPU type, memory, timeouts)
- The CLI entry point `astrocytes` maps to `astrocytes:main` but the `main()` function implementation is not shown in current code
- Extensive TODO comments indicate active development
- Schema design prioritizes metadata preservation (mouse_id, slice_id, date_acquired, etc.)
