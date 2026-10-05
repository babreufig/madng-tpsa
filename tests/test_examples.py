# tests/test_examples.py

import runpy
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parents[1] / 'examples').glob('[0-9][0-9]_*.py'))


@pytest.mark.parametrize('example', EXAMPLES, ids=lambda path: path.name)
def test_example_runs(example):
    try:
        runpy.run_path(str(example), run_name='__main__')
    except ModuleNotFoundError:
        pytest.skip(f'Skipping {example} due to missing IPython')
