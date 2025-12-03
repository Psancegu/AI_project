# QLearningStructure.md

## State: tuple of ints that represent discretized states of the portfolio and the index

- % of cash in stocks (Low = 0, Mid = 1, High = 2)
- portfolio profitability in the last 30 days (Negative = -1, Neutral = 0, Positive = 1)
- profitablity of the index of reference of the project in the las 30 days (Negative = -1, Neutral = 0, Positive = 1)
- portfolio volatility (calculated with std deviation) --> $ \sigma = $\sigma = \sqrt{\frac{\sum_{i=1}^N (x_i - \mu)^2}{N}} $
- Concentration of portfolio (dependency on few stocks) --> (>N = 1,<=N = 0)


## Actions 
- Hold (Do nothing)
- Buy 5% of the remaining cash in the top performance stock
- Buy 10% of the remaining cash in the top-5 performance stocks equally
- Sell 5% of the portfolio using the worst performing stocks
- Sell 10% of the portfolio using the worst performing stocks 
- Diversificate (Redistribute all stocks in portfolio, function pending to decide process)
- Concentrate (Redistribute all stocks in portfolio, funcion pendin to decide process)
- Sell All (sell 100% of the portfolio)
- Buy All (Buy 100% of the cas remaining in the topn stocks)


## Reward Function
- Pending to develop cause of complexity and difficulty to integrate all causes and action into it.
