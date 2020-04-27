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
    global_aggregate: Location = Location(1, 'Global')
    usa: Location = Location(102, 'United States of America')
    washington: Location = Location(570, 'Washington')
    life_care: Location = Location(60406, 'Life Care Center, Kirkland, WA')
    king_and_snohomish: Location = Location(60405, 'King and Snohomish Counties (excluding Life Care Center), WA')
    other_wa_counties: Location = Location(60407, 'Other Counties, WA')


LOCATIONS = _Locations()


class _Columns(NamedTuple):
    location_id: str = 'location_id'
    parent_id: str = 'parent_id'
    location: str = 'Location'
    location_bad: str = 'location'
    location_name: str = 'location_name'
    country: str = 'Country/Region'
    state: str = 'Province/State'
    level: str = 'level'

    date: str = 'Date'
    days: str = 'Days'
    day1: str = 'Day1'
    last_day: str = 'last_day'
    last_day_two: str = 'last_day_two'
    two_date: str = 'two_date'
    threshold_date: str = 'threshold_date'

    population: str = 'population'

    age_group: str = 'age_group'
    age_group_weight: str = 'age_group_weight_value'

    deaths: str = 'Deaths'
    death_rate: str = 'Death rate'
    implied_death_rate: str = 'Implied death rate'
    death_rate_bad: str = 'death_rate'
    age_standardized_death_rate: str = 'Age-standardized death rate'
    ln_age_death_rate: str = 'ln(age-standardized death rate)'
    obs_ln_age_death_rate: str = 'Observed ln(age-standardized death rate)'
    ln_death_rate: str = 'ln(death rate)'
    delta_ln_asdr: str = 'Delta ln(asdr)'
    observed_delta_ln_asdr: str = 'Observed delta ln(asdr)'
    first_point: str = 'first_point'
    last_point: str = 'last_point'

    confirmed: str = 'Confirmed'
    confirmed_case_rate: str = 'Confirmed case rate'

    pseudo: str = 'pseudo'


COLUMNS = _Columns()
