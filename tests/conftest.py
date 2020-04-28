from pathlib import Path

import pandas as pd
import pytest


def _load(file):
    data = pd.read_csv(Path(__file__).absolute().parent / 'fixtures' / file)
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
    return data


@pytest.fixture
def age_death_df() -> pd.DataFrame:
    return _load('age_death_df.csv')


@pytest.fixture
def age_pop_df() -> pd.DataFrame:
    return _load('age_pop_df.csv')


@pytest.fixture
def age_standardized() -> pd.DataFrame:
    return _load('age_standardized.csv')


@pytest.fixture
def backcast() -> pd.DataFrame:
    df = _load('backcast.csv')
    df['last_day_two'] = pd.to_datetime(df['last_day_two'])
    df['two_date'] = pd.to_datetime(df['two_date'])
    return df


@pytest.fixture
def death_df() -> pd.DataFrame:
    return _load('death_df.csv')


@pytest.fixture
def get_standard_age_death_df() -> pd.DataFrame:
    return _load('get_standard_age_death_df.csv')


@pytest.fixture
def implied_df() -> pd.DataFrame:
    return _load('implied_df.csv')


@pytest.fixture
def process_death_df() -> pd.DataFrame:
    return _load('process_death_df.csv')



