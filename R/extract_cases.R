# TODO: Find out if this is actually being used.
library(data.table)

datestamp <- "2020_03_30"


# get date information
data_all <- read.csv(sprintf("/ihme/code/dbd/hkl1/covid_19_model/model_data/state_data_%s.csv", datestamp))
data_all$date <- as.Date(data_all$date, "%Y - %m -%d")
locations <- as.character(unique(data_all$location))
AgeBins <- c(0,55,65,75,85,100)
day_start <- min(data_all$date) - 20
day_end <- max(data_all$date) + 20
Day_vec <- seq(day_start,day_end, by = "1 day")
Days <- as.character(Day_vec)

# store csvs of draws
for (location in locations) {
  print(location)
  tmpCases <- readRDS(sprintf("/ihme/covid-19/hospitalizations/output/%s/%s/cases.RDS", datestamp, location))
  CasesArray <- array(0, dim=c(length(tmpCases), 5, length(Days)))
  for (i in 1:length(tmpCases)){
    CasesArray[i,,] <- tmpCases[[i]]
  }

  # agg ages
  AllAgeCasesArray <- array(0, dim=c(length(tmpCases), length(Days)))
  for (i in 1:length(tmpCases)){
    AllAgeCasesArray[i,] <- apply(CasesArray[i,,], 2, sum)
  }

  dt <- data.table(t(AllAgeCasesArray))
  setnames(dt, paste0("draw_", 0:999))
  dt$date <- Days
  dt$location <- location
  write.csv(dt[, c("location", "date", paste0("draw_", 0:999))],
            sprintf("/ihme/covid-19/deaths/counterfactual/cases_%s/%s.csv", datestamp, location))
}

