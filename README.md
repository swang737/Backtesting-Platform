# Backtesting-Platform Documentation
## DataLoader
DataLoader takes in a 2D array of prices per day that is (time x prices)

### Initiation Arguments
Dataloader(data, stocks)
- **data (mat)** is given as 2D array of prices per day
- **stocks (arr)** is optional argument of list of stocks of interest

### Attributes
#### data (mat) 
##### attribute ahh
current price data stored in class
- nt (int): total timesteps of data
- nins (int): total instruments of data
- stocks (arr): current stocks class is using
- t (int): current timestep
