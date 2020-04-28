import pandas.testing as pdt
from covid_model_deaths import data


def test_compute_backast(death_df, age_pop_df, age_death_df, backcast):
    backcast_actual = data.compute_backcast_log_age_specific_death_rates(death_df, age_pop_df, age_death_df,
                                                                         555, subnat=True, rate_threshold=-15)
    pdt.assert_frame_equal(backcast, backcast_actual)


def test_moving_average(moving_average_input, moving_average_output):
    moving_average_actual = data.moving_average_log_age_standardized_death_ratio(moving_average_input, -15)
    pdt.assert_frame_equal(moving_average_output, moving_average_actual)
