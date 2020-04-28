import pandas.testing as pdt
from covid_model_deaths import data


def test_compute_backast(death_df, age_pop_df, age_death_df, backcast):
    backcast_actual = data.compute_backcast_log_age_specific_death_rates(death_df, age_pop_df, age_death_df,
                                                                         555, subnat=True, rate_threshold=-15)
    pdt.assert_frame_equal(backcast, backcast_actual, check_like=True)
