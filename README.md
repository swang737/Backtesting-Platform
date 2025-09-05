# Backtesting-Platform Documentation
**WIP:** 
- Vector ARIMA modelling
- Strategy module
- Add to topCorr so that topCorr pairs are also produced (in general)

**Locally updated (not on repo):** 
- Changed stepTime to have option to return full price history (docs updated)
- plotPrices (docs updated)
- Added seasonality to ARIMA on EDA (docs updated)
- topCorr method to EDA (docs updated)
- getLaggedFeatures method to DataLoader

## DataLoader
Stock data manipulator. DataLoader takes in a 2D array of prices per day that is (time x prices).

### Initiation Arguments
Dataloader(data, stocks)
- **data (mat)** is given as 2D array of prices per day
- **stocks (arr)** is optional argument of list of stocks of interest

### Attributes
- **data(mat):** current price data stored in class
- **nt (int):** total timesteps of data
- **nins (int):** total instruments of data
- **stocks (arr):** current stocks class is using
- **t (int):** current timestep

### Navigation Methods
- **resetTime:** resets current timestep to 0
- **goToTime:** goes to specified timestep
- **goToEnd:** goes to last timestep
- **currentTime:** gets current timestep
- **stepTime:** returns current prices and steps forward 1 timestep
    - **history (bool):** False to use only today's returns. Default is full price history from today
    - **step (int):** can be specified as argument (number of timesteps forward)

### Main Methods
- **lookBack:** returns prices of every day up to specified amount of timesteps back
    - **t (int):** number of timesteps to look back (outputted num of arrays)
- **currentPrices:** returns prices at current timestep
- **getReturns:** returns todays returns (yesterday to today):
    - **log (bool):** if false, then calculates the simple returns, but default is log returns.
- **returnsToNow:** all returns until today
    - **log (bool):** if false, then calculates the simple returns. Default is log returns
- **lookBackReturns:** returns t number of returns in the past (all together)
    - **t (int):** number of timesteps to look back (outputted num of arrays)
    - **log (bool):** if false, then calculates the simple returns, but default is log returns.

# EDA
Exporatory Data Analysis tool. EDA takes in a DataLoader object ONLY.

## Initation Arguments
EDA(loader)
- **loader(obj):** DataLoader object

## Attributes
- **loader(obj)**: the DataLoader object associated with the class

## Methods
- **plotReturns:** plots returns timeseries of all stocks
    - **log (bool):** False to use simple returns, default is log returns.
    - **ncols (int):** Number of columns in plot matrix, default is 1
- **plotPrices:** plots prices timeseries of all stocks on the same graph
- **topCorrelated:** outputs correlations of all stocks with specified stock in descending order (RETURNS ONLY for now)
    - **stocknum (int):** specify stock number to look at correlation for
    - **dropSelf (bool):** drops the first correlation (usually itself). Default is False, just if presenting :)
    - **log (bool):** uses log returns, default is True, but False uses simple returns.
- **getACF:** returns the Autocorrelation Function of all lags for all stocks (stocks x lags matrix) AND plots all of them
    - **save (bool):** saves graphs as png to Notebooks folder, default is false
    - **graph (bool):**  plots ACF for all stocks, default is true
    - **lag (int):** how many lags to look back, default is 20
    - **ncols (int):** how many columns to do for graphing, default is 3
    - **conf (bool):** whether or not to include conf interval (anything below is noise)
- **getPACF:** returns the Partial Autocorrelation Function of all lags for all stocks (stocks x lags matrix) AND plots all of them
    - **save (bool):** saves graphs as png to Notebooks folder, default is false
    - **graph (bool):**  plots PACF for all stocks, default is true
    - **lag (int):** how many lags to look back, default is 20
    - **ncols (int):** how many columns to do for graphing, default is 3
    - **conf (bool):** whether or not to include conf interval (anything below is noise)
- **getADF:** returns Augmented Dickey-Fuller statistics -> Check for weak stationality
    - **log (bool)**: False to use simple returns. Default is log returns.
- **fitARIMA:** fits ARIMA model to all stocks.
    - **params (arr):** parameters in the form of [p, d, q]
    - **seasonal_params (arr):** parameters in the form of [p, d, q, n], leave blank if not seasonal.
    - **train_len:** can either be ratio or integer of training set length
    - **warning (bool):** True to show warnings. Default is False.
    - **save (bool):** False to not save image. Default is True.
    - **ncols (int):** Number of columns in plot matrix, default is 1
    - **log (bool):** False to use simple returns, default is log returns.
- **backtestARIMA:** backtests ARIMA model, outputs MAE, RMSE and plots rolling forecasts
    - **params (arr):** parameters in the form of [p, d, q]
    - **seasonal_params (arr):** parameters in the form of [p, d, q, n], leave blank if not seasonal.
    - **train_len:** can either be ratio or integer of training set length
    - **warning (bool):** True to show warnings. Default is False
    - **save (bool):** False to not save image. Default is True
    - **ncols (int):** number of columns in plot matrix, default is 1.
    - **log (bool):** false to use simple returns, default is log returns.


