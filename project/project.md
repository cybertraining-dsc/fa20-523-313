# Analyzing LSTM Performance on Predicting Stock Market for Multiple Time Steps

[![Check Report](https://github.com/cybertraining-dsc/fa20-523-313/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-313/actions)
[![Status](https://github.com/cybertraining-dsc/fa20-523-313/workflows/Status/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-313/actions)
Status: in progress



Fauzan Isnaini, [fa20-523-313](https://github.com/cybertraining-dsc/fa20-523-313/), [Edit](https://github.com/cybertraining-dsc/fa20-523-313/blob/main/project/project.md)

{{% pageinfo %}}

## Abstract

Predicting the stock market has been an attractive field of research for a long time because it promises big wealth for anyone who can find the secret. For a long time, traders around the world have been relying on technical analysis to analyze patterns in the stock price movement and predict the trend. With the advancement of big data, some financial institutions are beginning to predict the market by creating a model of the market using machine learning. While some researches produce promising results, most of them are directed on predicting the next day market behavior. In this study, we created an LSTM model to predict the market for multiple time frames. We then analyzed the performance of the model for different time period. From our observations, LSTM is good at predicting 5 time steps ahead, but the prediction became inaccurate as the time frame gets longer. 



Contents

{{< table_of_contents >}}

{{% /pageinfo %}}

**Keywords:** stock, market, predictive analytics, LSTM, random forest, regression, technical analysis 


## 1. Introduction

Stock market prediction is a fascinating field of study for many analysts and researchers because it can give a significant amount of money. While there are numerous studies conducted in this field, predicting the stock market remains a challenging task, because of its noisy and non-stationary nature [^1]. The stock market is "noisy", because it is sensitive with the mass psychology. The trends and patterns in stock market can also change abruptly because of bad news, natural disasters, and some unforeseen circumstances, thus it is considered non-stationary.
The efficient market hypothesis even suggests that predicting or forecasting the financial market is unrealistic, because price changes in the real world are unpredictable. All the changes in prices of the financial market are based on immediate economic events or news. Investors are profit-oriented, their buying or selling decisions are made according to most recent events regardless past analysis or plans. The argument about this Efficient Market Hypothesis has never been ended. So far, there is no strong proof that can verify if the efficient market hypothesis is proper or not [^2]. 
However, as Yaser [^3] claims, financial markets are predictable to a certain extent. The past experience of many price changes over a certain period of time in the financial market and the undiscounted serial correlations among vital economic events affecting the future financial market are two main pieces of evidence opposing the Efficient Market Hypothesis.
The most popular methods in predicting the stock markets are technical and fundamental analysis. Fundamental analysis is mainly based on three essential aspects [^4]: (i) macroeconomic analysis such as Gross Domestic Products and Consumer Price Index (CPI) which analyses the effect of the macroeconomic environment on the future profit of a company, (ii) industry analysis which estimates the value of the company based on industry status and prospect, and (iii) company analysis which analyses the current operation and financial status of a company to evaluate its internal value. 
On the other hand, technical analysis is grouped into eight domains [^4]: sentiment, flow-of-funds, raw data, trend, momentum, volume, cycle, and volatility. Sentiment represents the behaviours of various market participants. Flow-of-funds is a type of indicator used to investigate the financial status of various investors to pre-evaluate their strength in terms of buying and selling stocks, then, corresponding strategies, such as short squeeze can be adopted. Raw data include stock price series and price patterns such as K-line diagrams and bar charts. Trend and momentum are examples of price-based indicators, trend is used for tracing the stock price trends while momentum is used to evaluate the velocity of the price change and judge whether a trend reversal in stock price is about to occur. Volume is an indicator that reflects the enthusiasm of both buyers and sellers for investing, it is also a basis for predicting stock price movements. The cycle is based on the theory that stock prices vary periodically in the form of a long cycle of more than 10 years containing short cycles of a few days or weeks. Finally, volatility is often used to investigate the fluctuation range of stock prices and to evaluate risk and identify the level of support and resistance.
While those two are still the most popular approaches, the age of big data has brought a new method to predict the stock market: quantitative analysis. In this new method, stock market is captured into a mathematical model, and machine learning is used to predict its behavior.Research by Alzazah and Cheng [^5] analyzed more than 50 articles to compare various machine learning (ML) and deep learning (DL) methods used  to find which method could be more effective in prediction and for which types and amount of data. This research has proven that quantitative analysis with LSTM gives a promising result as the predictor of a stock market.
In this study, we analyzed the performance of LSTM in predicting the stock market for multiple time frames. Despite the promising result in LSTM, most of the previous studies are conducted in building a model to predict the next day price, thus, we wanted to know whether LSTM can predict for a longer time frame.

## 2. Background Research and Previous Work


### 2.2 Predicting the Next Recession using Long Short-term Memory (LSTM) Algorithm

Another long term prediction in this area is from Khedkar [^7]. He used LSTM in predicting the next recession in India. In this study, the stock closing price data is used, instead of financial data – which is technical analysis. LSTM networks are well-suited to classifying, processing, and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series. The study managed to predict the stock market crash in 2020.

### 2.3 Stock Market Prediction Using Text Mining

Text mining usually accompanies the technical analysis method. We can conduct sentiment analysis on news and social media to predict whether the stock market goes up or down. Sentiments on Twitter usually indicate market volatility and volume, while sentiments on news are a better predictor of stock movement [4].

## 3. Choice of Data-sets

In predicting the outcome, these datasets will be utilized:
Yahoo Finance (finance.yahoo.com). Yahoo Finance contains a lot of both fundamental and technical data, and they are free of charge.
Twitter (twitter.com). Sentiment analysis on Twitter will be conducted to represent the mass psychology regarding the economic condition in Indonesia

## 4. Methodology

In this study, we will predict market movement in the next two years using technical approach accompanied by sentiment analysis. While fundamental approach is a strong predictor of the long term stock price movement for an individual stock, it is very difficult to utilize it in an index like IDX composite because it will takes too many variables. The technical approach will utilize LSTM, which has been widely used in time series analysis.
While there are previous works on predicting stock markets, most of them are based on 60 days sliding windows to predict the price one day ahead. While this approach give a good result for a single time-step analysis, there is no evidence that this approach can work fine for multiple time-steps.

## 5. Inference

This section will be addressed upon project completion.

## 6. Conclusion

This section will be addressed upon project completion.

## 7. Acknowledgements

The author would like to thank Dr. Geoffrey Fox, Dr. Gregor Von Laszewski, and the associate instructors in the FA20-BL-ENGR-E534-11530: Big Data Applications course (offered in the Fall 2020 semester at Indiana University, Bloomington) for their continued assistance and suggestions concerning exploring this idea and also for their aid with preparing the various drafts of this article.

## 8. References

[^1]: A. Nikfarjam, E. Emadzadeh, and S. Muthaiyah, "Text mining approaches for stock market prediction," 2010 The 2nd International Conference on Computer and Automation Engineering (ICCAE), 2010.

[^2]: A. Singh, "Stock Price Prediction Using Machine Learning: Deep Learning, " Analytics Vidhya, 18-Oct-2020. [Online]. Available: <https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/>. [Accessed: 20-Oct-2020].


[^3]: C. Stoean, W. Paja, R. Stoean, and A. Sandita, "Deep architectures for long-term stock price prediction with a heuristic-based strategy for trading simulations," Plos One, vol. 14, no. 10, 2019.

[^4]: F. S. Alzazah and X. Cheng, "Recent Advances in Stock Market Prediction Using Text Mining: A Survey," E-Business [Working Title], 2020.

[^5]: N. Chauhan, "Stock Market Forecasting Using Time Series Analysis," KDnuggets, 2020. [Online]. Available: <https://www.kdnuggets.com/2020/01/stock-market-forecasting-time-series-analysis.html>. [Accessed: 20-Oct-2020].

[^6]: N. Milosevic, "Equity forecast: Predicting long term stock price movement using machine learning," 2018.

[^7]: S. Khedkar, "Stock Market Prediction Using Deep Learning and Python," Medium, 27-Sep-2019. [Online]. Available: <https://medium.com/analytics-vidhya/stock-market-prediction-using-python-article-4-the-next-recession-923185a2736f>. [Accessed: 20-Oct-2020].

