# ============================================================
# TIME SERIES FORECASTING WITHOUT TRAIN-TEST SPLIT
# Change ONLY target_column
# ============================================================

library(forecast)
library(ggplot2)
library(dplyr)

# 1. LOAD DATA

df <- read.csv(
  "D:\\data analyst\\R_yt\\sugercane\\python_ethanol\\forecasting_dataset_intern_final.csv",
  sep = ","
)

dim(df)
str(df)
#View(df)
df <- na.omit(df)




# ------------------------------------------------------------
# 1. SELECT TARGET VARIABLE
# ------------------------------------------------------------

target_column <- "Maize_production_lakh.tonnes"


# ------------------------------------------------------------
# 2. PREPARE DATA
# ------------------------------------------------------------

ts_data <- df %>%
  select(year, all_of(target_column)) %>%
  filter(!is.na(.data[[target_column]])) %>%
  arrange(year)

target_ts <- ts(
  ts_data[[target_column]],
  start = min(ts_data$year),
  frequency = 1
)


# ------------------------------------------------------------
# 3. PLOT ORIGINAL TIME SERIES
# ------------------------------------------------------------

ggplot(
  ts_data,
  aes(x = year, y = .data[[target_column]])
) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  labs(
    title = paste(target_column, "vs Year"),
    x = "Year",
    y = target_column
  ) +
  theme_minimal()


# ------------------------------------------------------------
# 4. ACF AND PACF
# ------------------------------------------------------------

par(mfrow = c(1, 2))

acf(
  target_ts,
  main = paste("ACF -", target_column)
)

pacf(
  target_ts,
  main = paste("PACF -", target_column)
)

par(mfrow = c(1, 1))


# ------------------------------------------------------------
# 5. LJUNG-BOX AUTOCORRELATION TEST
# ------------------------------------------------------------

lag_value <- min(
  10,
  floor(length(target_ts) / 3)
)

ljung_test <- Box.test(
  target_ts,
  lag = lag_value,
  type = "Ljung-Box"
)

print(ljung_test)

if(ljung_test$p.value < 0.05) {
  print("Autocorrelation is present")
} else {
  print("No significant autocorrelation detected")
}


# ============================================================
# 6. MOVING AVERAGE
# ============================================================

max_window <- min(
  10,
  floor(length(target_ts) / 3)
)

ma_results <- data.frame(
  Window = integer(),
  MAPE = numeric()
)


for(w in 2:max_window) {
  
  ma_fit <- ma(
    target_ts,
    order = w
  )
  
  valid <- !is.na(ma_fit)
  
  actual <- as.numeric(target_ts)[valid]
  
  predicted <- as.numeric(ma_fit)[valid]
  
  
  mape <- mean(
    abs(
      (actual - predicted) / actual
    )
  ) * 100
  
  
  ma_results <- rbind(
    ma_results,
    data.frame(
      Window = w,
      MAPE = mape
    )
  )
}


print(ma_results)


# Find Best Window

best_window <- ma_results$Window[
  which.min(ma_results$MAPE)
]

cat(
  "\nBest Moving Average Window:",
  best_window,
  "\n"
)


# Best Moving Average Model

ma_model <- ma(
  target_ts,
  order = best_window
)


# Moving Average MAPE

valid <- !is.na(ma_model)

ma_actual <- as.numeric(target_ts)[valid]

ma_predicted <- as.numeric(ma_model)[valid]


ma_mape <- mean(
  abs(
    (ma_actual - ma_predicted) /
      ma_actual
  )
) * 100


cat(
  "\nMoving Average MAPE:",
  round(ma_mape, 2),
  "%\n"
)


# ------------------------------------------------------------
# MOVING AVERAGE PLOT
# ------------------------------------------------------------

ma_plot_data <- data.frame(
  
  year = ts_data$year[valid],
  
  Actual = ma_actual,
  
  Moving_Average = ma_predicted
  
)


ggplot(
  ma_plot_data,
  aes(x = year)
) +
  
  geom_line(
    aes(y = Actual, colour = "Actual"),
    linewidth = 1
  ) +
  
  geom_line(
    aes(
      y = Moving_Average,
      colour = "Moving Average"
    ),
    linewidth = 1
  ) +
  
  geom_point(
    aes(y = Actual, colour = "Actual")
  ) +
  
  labs(
    title = paste(
      "Moving Average Model - Window",
      best_window
    ),
    x = "Year",
    y = target_column,
    colour = "Series"
  ) +
  
  theme_minimal()


# ============================================================
# 7. SINGLE EXPONENTIAL SMOOTHING
# ============================================================

ses_model <- ses(
  target_ts,
  h = 3
)


print(ses_model$model)


# Get Fitted Values

ses_fitted <- fitted(
  ses_model$model
)


# Calculate SES MAPE

ses_accuracy <- accuracy(
  ses_model$model
)

print(ses_accuracy)


ses_mape <- ses_accuracy[
  "Training set",
  "MAPE"
]


cat(
  "\nSingle Exponential Smoothing MAPE:",
  round(ses_mape, 2),
  "%\n"
)


# ------------------------------------------------------------
# SES PLOT
# ------------------------------------------------------------

autoplot(target_ts) +
  
  autolayer(
    ses_fitted,
    series = "SES Fitted"
  ) +
  
  labs(
    title = paste(
      "Single Exponential Smoothing -",
      target_column
    ),
    x = "Year",
    y = target_column
  ) +
  
  theme_minimal()


# ============================================================
# 8. ARIMA
# ============================================================

arima_model <- auto.arima(
  target_ts
)


print(arima_model)


# ------------------------------------------------------------
# PRINT ARIMA PARAMETERS
# ------------------------------------------------------------

arima_order <- arimaorder(
  arima_model
)

cat(
  "\nARIMA PARAMETERS\n"
)

cat(
  "p =",
  arima_order["p"],
  "\n"
)

cat(
  "d =",
  arima_order["d"],
  "\n"
)

cat(
  "q =",
  arima_order["q"],
  "\n"
)


# ------------------------------------------------------------
# ARIMA FITTED VALUES
# ------------------------------------------------------------

arima_fitted <- fitted(
  arima_model
)


# ARIMA ACCURACY

arima_accuracy <- accuracy(
  arima_model
)

print(arima_accuracy)


arima_mape <- arima_accuracy[
  "Training set",
  "MAPE"
]


cat(
  "\nARIMA MAPE:",
  round(arima_mape, 2),
  "%\n"
)


# ------------------------------------------------------------
# ARIMA FITTED PLOT
# ------------------------------------------------------------

autoplot(target_ts) +
  
  autolayer(
    arima_fitted,
    series = "ARIMA Fitted"
  ) +
  
  labs(
    title = paste(
      "ARIMA Model Fit -",
      target_column
    ),
    x = "Year",
    y = target_column
  ) +
  
  theme_minimal()


# ------------------------------------------------------------
# ARIMA RESIDUAL CHECK
# ------------------------------------------------------------

checkresiduals(
  arima_model
)


# ============================================================
# 9. COMPARE MODEL MAPE
# ============================================================

comparison_matrix <- data.frame(
  
  Model = c(
    "Moving Average",
    "Single Exponential Smoothing",
    "ARIMA"
  ),
  
  MAPE = c(
    ma_mape,
    ses_mape,
    arima_mape
  )
  
)


comparison_matrix <- comparison_matrix %>%
  arrange(MAPE)


print(
  comparison_matrix
)


# ------------------------------------------------------------
# PRINT BEST MODEL
# ------------------------------------------------------------

best_model <- comparison_matrix$Model[1]

best_mape <- comparison_matrix$MAPE[1]


cat(
  "\nBEST MODEL:",
  best_model,
  "\n"
)

cat(
  "LOWEST MAPE:",
  round(best_mape, 2),
  "%\n"
)


# ============================================================
# 10. MAPE COMPARISON GRAPH
# ============================================================

ggplot(
  comparison_matrix,
  aes(
    x = reorder(Model, MAPE),
    y = MAPE
  )
) +
  
  geom_col() +
  
  geom_text(
    aes(
      label = paste0(
        round(MAPE, 2),
        "%"
      )
    ),
    vjust = -0.5
  ) +
  
  labs(
    title = paste(
      "In-Sample Model Accuracy Comparison -",
      target_column
    ),
    x = "Model",
    y = "MAPE (%)"
  ) +
  
  theme_minimal()

