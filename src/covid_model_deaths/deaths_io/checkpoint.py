"""Check-pointing of model progress by writing intermediate outputs."""
from pathlib import Path
from typing import Any, List, Union

from covid_shared.shell_tools import mkdir
import dill as pickle
from loguru import logger


class Checkpoint:
    """Abstraction for check-pointing model progress."""

    def __init__(self, output_root: Union[str, Path], clear: bool = False):
        self.checkpoint_dir = self._setup_checkpoint_dir(output_root, clear)
        self.cache = {}
        self._io = _PickleIO()

    @property
    def keys(self):
        return self._io._keys(self.checkpoint_dir)

    def write(self, key, data):
        if key in self.cache:
            logger.warning(f"Overwriting {key} in checkpoint data.")
        self.cache[key] = data
        self._io._write(self.checkpoint_dir, key, data)

    def load(self, key):
        if key in self.cache:
            logger.info(f'Loading {key} from in memory cache.')
        else:
            logger.info(f'Reading {key} from checkpoint dir {self.checkpoint_dir}.')
            self.cache[key] = self._io._load(self.checkpoint_dir, key)
        return self.cache[key]

    @staticmethod
    def _setup_checkpoint_dir(output_root: Union[str, Path], clear: bool) -> Path:
        checkpoint_dir = Path(output_root) / 'checkpoint'
        if clear and checkpoint_dir.exists():
            logger.debug(f'Clearing previous checkpoint data.')
            for p in checkpoint_dir.iterdir():
                p.unlink()
            checkpoint_dir.rmdir()

        logger.debug(f'Making checkpoint directory at {str(checkpoint_dir)}')
        mkdir(checkpoint_dir, exists_ok=True)
        return checkpoint_dir

    def __repr__(self):
        return f'Checkpoint({str(self.checkpoint_dir)})'


class _PickleIO:
    """Reads and writes pickles."""

    @staticmethod
    def _keys(root: Path) -> List[str]:
        return [p.stem for p in root.iterdir() if p.suffix == '.pkl']

    @staticmethod
    def _write(root: Path, key: str, data: Any):
        path = root / f'{key}.pkl'
        with path.open('wb') as data_file:
            pickle.dump(data, data_file, -1)

    @staticmethod
    def _load(root: Path, key: str) -> Any:
        path = root / f'{key}.pkl'
        with path.open('rb') as data_file:
            data = pickle.load(data_file)
        return data
