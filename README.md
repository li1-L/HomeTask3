# HomeTask3
- Simple News-Driven Trading System
## Assignment Requirements
1. Download historical price data for at least one year of a selected asset (Bitcoin BTC-USD is adopted in this project).
2. Collect authorized public news resources related to the target asset.
3. Perform sentiment analysis on each news article to generate three trading instructions: Buy, Sell (close position), Hold.
4. Calculate the annual profit rate in percentage based on the initial investment capital.
5. Compare the performance of the news trading strategy with the traditional Buy & Hold strategy.

## Data Source Description
1. Price Data: Obtained from Yahoo Finance via yfinance module. If API access is rate-limited, simulated BTC price data with realistic volatility will be used automatically.
2. News Data: Real public news from Reuters and Bloomberg, embedded offline without API registration or network proxy.
3. Sentiment Model: VADER, a sentiment analysis model optimized for financial texts, outputs sentiment score ranging from -1 to +1.
4. Signal Threshold Rule: Score ≥ 0.2 → Buy; Score ≤ -0.2 → Sell; Other values → Hold

## Environment & Dependencies Installation
Run this command in CMD to install required libraries:
```bash
pip install yfinance pandas numpy matplotlib nltk
