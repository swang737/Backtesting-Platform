# Backtesting-Platform Documentation
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
- **currentTime:** gets current timestep
- **stepTime:** returns current prices and steps forward 1 timestep
    - **step (int):** can be specified as argument (number of timesteps forward)

### Main Methods
- **lookBack:** returns prices of every day up to specified amount of timesteps back
    - **t (int):** number of timesteps to look back (outputted num of arrays)
- **currentPrices:** returns prices at current timestep
- **getReturns:** returns todays returns (yesterday to today):
    - **log (bool):** if false, then calculates the simple returns, but default is log returns.
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
- **getACF:** returns the Autocorrelation Function of all lags for all stocks (stocks x lags matrix) AND plots all of them
    - **save (bool):** saves graphs as png to Notebooks folder, default is false
    - **graph (bool):**  plots ACF for all stocks, default is true
    - **lag (int):** how many lags to look back, default is 20
    - **ncols (int):** how many columns to do for graphing, default is 3
    - **conf (bool):** whether or not to include conf interval (anything below is noise)


