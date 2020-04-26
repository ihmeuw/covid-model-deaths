"""Abstraction to consistently manage input data."""
from pathlib import Path
from typing import NamedTuple, Union

from loguru import logger
import pandas as pd


class _Measures(NamedTuple):
    age_death: str = 'age_death.csv'
    age_pop: str = 'age_pop.csv'
    confirmed: str = 'confirmed.csv'
    deaths: str = 'deaths.csv'
    full_data: str = 'full_data.csv'
    us_pops: str = 'us_pops.csv'


MEASURES = _Measures()


class InputsContext:
    """Manages access to input data."""

    def __init__(self, inputs_root: Union[str, Path]):
        self.root = Path(inputs_root).resolve()

    def load(self, measure: str) -> pd.DataFrame:
        """Loads an available measure from the input root.

        Parameters
        ----------
        measure
            The measure file to load.

        Returns
        -------
            The data associated with the provided measure.

        """
        logger.debug(f"Loading {measure} from {str(self.root)}.")
        if measure not in MEASURES:
            raise ValueError(f'Invalid measure {measure} - valid measures are {", ".join(MEASURES)}.')

        path = self.root / measure

        if path.suffix == '.csv':
            data = pd.read_csv(path)
        elif path.suffix == '.xlsx':
            data = pd.read_excel(path)
        else:
            raise ValueError(f'Unknown file type {path.suffix}.')

        data = self._clean_columns(data)
        data = self._validate(measure, data)

        return data

    @staticmethod
    def _clean_columns(data: pd.DataFrame) -> pd.DataFrame:
        """Standardizes input data columns."""
        if 'date' in data.columns:
            data.loc[:, 'date'] = pd.to_datetime(data['date'])

        if 'Date' in data.columns:
            data.loc[:, 'Date'] = pd.to_datetime(data['Date'])

        if 'location_id' in data.columns:
            data.loc[:, 'location_id'] = data['location_id'].astype(int)

        return data

    @staticmethod
    def _validate(measure: str, data: pd.DataFrame):
        """Ensure data meets all quality standards."""
        # TODO: check data preconditions.
        return data
