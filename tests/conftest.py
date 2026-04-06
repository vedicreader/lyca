"""Pytest configuration for lyca tests.

Automatically downloads the test model if LYCA_MODEL_PATH is not set
but network is available. Set LYCA_MODEL_PATH to a .litertlm file to
skip download.
"""

import os
import pytest


def pytest_configure(config):
    """Try to download model if LYCA_MODEL_PATH not set."""
    if os.environ.get('LYCA_MODEL_PATH'):
        return
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id='litert-community/gemma-4-E2B-it-litert-lm',
            filename='gemma-4-E2B-it.litertlm'
        )
        os.environ['LYCA_MODEL_PATH'] = path
        print(f'\n✓ Model auto-downloaded to: {path}')
    except Exception as e:
        print(f'\n⚠ Model download failed ({e}). Model-dependent tests will be skipped.')
        print('  Set LYCA_MODEL_PATH=/path/to/gemma-4-E2B-it.litertlm to run all tests.')
