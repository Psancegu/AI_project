# Reinforcement Learning Structure for Trading Agent

## State (Cash, Trend, Volatility, Performance)

We define the state as a tuple of 4 integers that rank from 0 to 2.
Example: (0,1,0,2)

Each value indicates:
- Cash $ \rightarrow $ (0,1,2) $ \rightarrow $ Percentage of cash in portfolio
    - [0] Invested: < 10% cash in portfolio
    - [1] Balanced: 10 - 50% cash in portfolio
    - [2] Capital: > 50% cash in portfolio


**Probably going to change trend to volatility indexes because it matters more and has a greater impact**

- Trend $ \rightarrow $ (0,1,2) $ \rightarrow $ Actual Price - SMA[1] 50d 
    - [0] Bearish: $ < -2 $%
    - [1] Neutral: $ \text{ in the range }\pm 2 $%
    - [2] Bullish: $ > +2 $%

- Relative Volatility $ \rightarrow $ (0,1) $ \rightarrow \sigma_{30d} \text{ vs } \sigma_{365d} $
    - [0] Normal: $ \sigma_{30d} \leq \sigma_{365d} $
    - [1] High Risk: $ \sigma_{30d} > \sigma_{365d} $

- Portfolio Performance $ \rightarrow $ (0,1,2) $ \rightarrow R_{portolio} - R_{index} $ in the last 30 days
    - [0] Underperforming: $ R_{portolio} < R_{index} $ 
    - [1] Neutral: $ R_{portolio} < R_{index} (\pm 0.5) $
    - [2] Overperforming: $ R_{portolio} > R_{index} $



## Actions 
- [0] Hold $ \rightarrow $ Do Nothing
- [1] Buy Conservative $ \rightarrow $ Buy 25% of remaining cash in stocks
- [2] Buy Agressive $ \rightarrow $ Buy 100% of remaining cash in stocks
- [3] Sell Conservative $ \rightarrow $ Sell 25% of actives to cash
- [4] Sell Agressive $ \rightarrow $ Sell 100% of actives to cash

**Buy Policy** $ \rightarrow $ Buy the Top5 performing stocks split equally

**Sell Policy** $ \rightarrow $ Sell the worst performing stock until we reach the cap

With this definition of the state and action the Q_table will have 270 values, 54 states x 5 actions.


## Reward Function
- With the specification above our reward function chosen for this project is:

$$
R_t = 
    \underbrace{
        \begin{cases} 
            r_t & \text{if } r_t \ge 0 \\ 
            1.5 \cdot r_t & \text{if } r_t < 0 
        \end{cases}
    }_{\text{Return with fear to loss}}
    - 
    \underbrace{
        \begin{cases} 
            0.1 & \text{if } A_t \neq 0 \\ 
            0 & \text{if } A_t = 0 
        \end{cases}
    }_{\text{Comissions}}
    + 
    \underbrace{
        \begin{cases} 
            0.04 & \text{if } \%_{cash} = 0 \ (<10\%) \\ 
            0.02 & \text{if } \%_{cash} = 1 \ (10\text{-}50\%) \\ 
            0 & \text{if } \%_{cash} = 2 \ (>50\%) 
        \end{cases}
    }_{\text{Active Portolio Reward}}
$$

We put a 1.5x multiplier to loss to make the agent panic with losses more than getting happy with profit. Then we put a 0.1 penalty to avoid the agent playing all day with buy/sell actions and losing all the money on comissions.
Finally we put a reward from 0 to 0.04 to enforce the agent to buy, and prevent to buy nothing always.


## Convergence

We will check convergence with the sum of absolute differences between the Q-Table at the end of the current episode and the previous episode.

## Data Split

To get the real accuracy of our model, we will divide the data into two datasets, similar to cross validation but respecting the time.

- Training data set will cover the period between January 1st 2000 and December 31st of 2021.
- Test data set will cover the period between January 1st 2022 and the last day possible to hand in the project.

---

[1]: Simple Moving Average, is the average of the last N values, when a new value enters, the oldes one gets out. Usually used in trading.
