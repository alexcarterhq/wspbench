from pathlib import Path
from types import SimpleNamespace

import pytest

from wspbench.runner import _check_frozen, freeze_protocol


def test_sanity_pool_cannot_be_frozen_or_run_as_headline(tmp_path, monkeypatch):
    import wspbench.tasks as tasks
    monkeypatch.setattr(tasks, 'load_tasks', lambda root, split: [SimpleNamespace(family='read_visual')])
    def unexpected_validation(root):
        raise AssertionError('Sanity gate should run before validation or paid preflight')
    monkeypatch.setattr(tasks, 'validate_suite', unexpected_validation)
    with pytest.raises(ValueError, match='sanity checks only'):
        freeze_protocol(tmp_path, tmp_path / 'runs', Path('configs/models.json'))
    with pytest.raises(ValueError, match='sanity checks only'):
        _check_frozen(tmp_path, tmp_path / 'runs', [], {})
