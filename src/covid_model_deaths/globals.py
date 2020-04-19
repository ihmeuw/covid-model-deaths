from typing import NamedTuple

LN_MORTALITY_RATE_THRESHOLD = -15
GBD_REPORTING_LOCATION_SET_ID = 35
GBD_2017_ROUND_ID = 6


MOBILITY_SOURCES = ['google', 'descartes', 'safegraph']
# TODO: Document better.  These are about the mix of social distancing
#  covariates.
COV_SETTINGS = [('equal', [1, 1, 1]),
                ('ascmid', [0.5, 1, 2]),
                ('ascmax', [0, 0, 1])]
# TODO: Don't know what this is at all. Something about days.
KS = [21]


class Location:
    """Simple wrapper for location data."""
    def __init__(self, loc_id: int, name: str):
        self.id = loc_id
        self.name = name

    def __repr__(self):
        return f'Location(id={self.id}, name={self.name})'


class _Locations(NamedTuple):
    usa: Location = Location(102, 'United States of America')
    washington: Location = Location(570, 'Washington')
    life_care: Location = Location(60406, 'Life Care Center, Kirkland, WA')
    king_and_snohomish: Location = Location(60405, 'King and Snohomish Counties (excluding Life Care Center), WA')
    other_wa_counties: Location = Location(60407, 'Other Counties, WA')



LOCATIONS = _Locations


class _Columns(NamedTuple):
    location_id: str = 'location_id'
    parent_id: str = 'parent_id'
    location: str = 'Location'
    location_bad: str = 'location'
    location_name: str = 'location_name'
    country: str = 'Country/Region'
    state: str = 'Province/State'

    date: str = 'Date'
    days: str = 'Days'
    last_day: str = 'last_day'
    threshold_date: str = 'threshold_date'

    population: str = 'population'

    deaths: str = 'Deaths'
    death_rate: str = 'Death rate'
    ln_age_death_rate: str = 'ln(age-standardized death rate)'
    ln_death_rate: str = 'ln(death rate)'

    confirmed: str = 'Confirmed'
    confirmed_case_rate: str = 'Confirmed case rate'

    pseudo: str = 'pseudo'

COLUMNS = _Columns()
