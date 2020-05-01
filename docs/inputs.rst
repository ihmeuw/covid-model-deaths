Input Overview
==============

.. list-table:: Inputs
   :widths: 25 25 25 25 25 50
   :header-rows: 1

   * - Input
     - Expected filepath
     - Created by
     - Versioned
     - Schema
     - Constraints
   * - age_death
     - /ihme/covid-19/model-inputs/{data_version}/age_death.csv
     - covid-input-etl
     - Versioned by etl tool
     - + location_id: Int. Location ids where we have an age-specific death
         pattern.
       + age_group: String. "X - Y" Where X and Y are age start and age end.
       + death_rate: Float. The age-specific mortality rate.  Looks annualized
         by the size of the numbers.
     - + location_id: One of [1, 68, 86, 102, 503].
       + age_group: 10 year increments starting at 0 except for the last bin
         which is "80-125".
       + death_rate: 0 <= value.
   * - age_pop
     - /ihme/covid-19/model-inputs/{data_version}/age_death.csv
     - covid-input-etl
     - Versioned by etl tool.
     - + location_id: Int. Location ids for all data in the JHU + David's data
         set. Includes some made up location ids that are not in a hierarchy
         yet.
       + age_group: String. "X - Y" Where X and Y are age start and age end.
       + population: Float. Count of population in the location and age group.
         Imputed, so there are fractional values.
       + age_group_weight_value: Float. Proportion of the population in the
         age group as compared to the total location population.
     - + location_id: Negative values indicated locations not in a
         hierarchy and should be flagged for Chantal.
       + age_group: 10 year increments starting at 0 except for the last bin
         which is "80-125".
       + population:  0 <= value.
       + age_group_weight_value: 0 <= value <= 1.
   * - confirmed
     - /ihme/covid-19/model-inputs/{data_version}/confirmed.csv
     - covid-input-etl
     - Versioned by etl tool.
     - + location_id: Int. Location ids for all data in the JHU + David's data
         set. Includes some made up location ids that are not in a hierarchy
         yet.
       + Location: String. The location name.  Previously (and still sometimes)
         use for location mapping in lieu of location ids.
       + Country/Region: String. The the country to which the location belongs.
         May have the same value as ``Location``
       + Date: String. The date of the observation.
       + Days: Int. Days since something.  TBD. **DOES NOT MATCH DEATHS**.
       + Deaths: Float (should be Int). Cumulative confirmed cases on the
         date. **MISLABELLED**
       + Death rate: Float. Cumulative confirmed case rate. **MISLABELLED**
       + population: Float. Total population for the location id.
     - + location_id: Negative values indicated locations not in a
         hierarchy and should be flagged for Chantal.
       + Location:
       + Country/Region:
       + Date: Format YYYY-MM-DD
       + Days: 0 <= value.  Strictly monotonic w/r/t Date
       + Deaths: 0 <= value. Weakly monotonic w/r/t Date **MISLABELLED**
       + Death rate: 0 <= value. Weakly monotonic w/r/t Date **MISLABELLED**
       + population: 0 <= value.
   * - deaths
     - /ihme/covid-19/model-inputs/{data_version}/deaths.csv
     - covid-input-etl
     - Versioned by etl tool.
     - + location_id: Int. Location ids for all data in the JHU + David's data
         set. Includes some made up location ids that are not in a hierarchy
         yet.
       + Location: String. The location name.  Previously (and still sometimes)
         use for location mapping in lieu of location ids.
       + Country/Region: String. The the country to which the location belongs.
         May have the same value as ``Location``
       + Date: String. The date of the observation.
       + Days: Int. Days since something.  TBD.  **DOES NOT MATCH CONFIRMED**
       + Deaths: Float (should be int). Cumulative deaths on the date.
       + Death rate: Float. Cumulative death rate.
       + population: Float. Total population for the location id.
     - + location_id: Negative values indicated locations not in a
         hierarchy and should be flagged for Chantal.
       + Location:
       + Country/Region:
       + Date: Format YYYY-MM-DD
       + Days: 0 <= value.  Strictly monotonic w/r/t Date
       + Deaths: 0 <= value. Weakly monotonic w/r/t Date
       + Death rate: 0 <= value. Weakly monotonic w/r/t Date
       + population: 0 <= value.
   * - full_data
     - /ihme/covid-19/model-inputs/{data_version}/full_data.csv
     - covid-input-etl
     - Versioned by etl tool.
     - + location_id: Int. Location ids for all data in the JHU + David's data
         set. Includes some made up location ids that are not in a hierarchy
         yet.
       + Province/State: String (nullable). The name of the location if
         the location is a subnational.  Otherwise null.
       + Country/Region: String. The the country to which the location belongs
         if a subnational.  Otherwise the name of the location.
       + Date: String. The date of the observation.
       + Confirmed: Float (should be Int). Cumulative cases confirmed on the
         Date.
       + Deaths: Float (should be Int). Cumulative deaths on the Date.
       + population: Float. Total population for the location id.
       + Confirmed case rate: Float. Cumulative case rate.
       + Death rate: Float.  Cumulative death rate.
     - + location_id: Negative values indicated locations not in a
         hierarchy and should be flagged for Chantal.
       + Province/State:
       + Country/Region:
       + Date: Format YYYY-MM-DD
       + Confirmed: 0 <= value. Weakly monotonic w/r/t Date
       + Deaths: 0 <= value. Weakly monotonic w/r/t Date
       + population 0 <= value.
       + Confirmed case rate: 0 <= value. Weakly monotonic w/r/t Date
       + Death rate: 0 <= value. Weakly monotonic w/r/t Date
   * - us_pops
     - /ihme/covid-19/model-inputs/{data_version}/us_pops.csv
     - covid-input-etl
     - Versioned by etl tool
     - + Province/State: String.  The name of the state.
       + age_group: 10 year increments starting at 0 except for the last bin
         which is "80-125".
       + population: Float. Total population for the state.
       + age_group_weight_value: Float.  Proportion of the total state
         population in the age group.
     - + Province/State:
       + age_group: 10 year increments starting at 0 except for the last bin
         which is "80-125".
       + population: 0 <= value.
       + age_group_weight_value: Float.  0 <= value <= 1.

