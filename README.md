# Backtesting-Platform Documentation
## DataLoader
DataLoader takes in a 2D array of prices per day that is (time x prices)

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

