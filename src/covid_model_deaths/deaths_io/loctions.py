from typing import NamedTuple

from db_queries import get_location_metadata
import pandas as pd


class _LocationGroups(NamedTuple):
    us: str = 'United States'
    non_us: str = 'Not United States'


LOCATION_GROUPS = _LocationGroups()


def get_locations(location_group: str, location_set_version_id: int) -> pd.DataFrame:
    loc_df = get_location_metadata(location_set_id=111,
                                   location_set_version_id=location_set_version_id)
    most_detailed = loc_df['most_detailed'] == 1
    good_ids = ~loc_df['location_id'].isin([53474, 53451, 53452])
    keep_columns = ['location_id', 'location_ascii_name', 'parent_id', 'level', 'most_detailed']

    loc_df = loc_df.loc[most_detailed * good_ids, keep_columns]
    parent_df = loc_df.loc[:, ['location_id', 'location_ascii_name']]
    parent_df = loc_df.rename(columns={'location_id': 'parent_id',
                                       'location_ascii_name': 'Country/Region'})

    loc_df = loc_df.merge(parent_df)

    if location_group == LOCATION_GROUPS.us:
        us = loc_df['path_to_top_parent'].str.startswith('102,')
        loc_df = loc_df.loc[us]
    elif location_group == LOCATION_GROUPS.non_us:
        not_us = ~loc_df['path_to_top_parent'].str.startswith('102,')
        loc_df = loc_df.loc[not_us]

    return loc_df.loc[:, ['location_id', 'Location', 'Country/Region', 'level']]

