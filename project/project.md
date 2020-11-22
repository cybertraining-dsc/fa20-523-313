# Predicting Stock Market Recovery After Pandemic

[![Check Report](https://github.com/cybertraining-dsc/fa20-523-313/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-313/actions)
[![Status](https://github.com/cybertraining-dsc/fa20-523-313/workflows/Status/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-313/actions)
Status: in progress



Fauzan Isnaini, [fa20-523-313](https://github.com/cybertraining-dsc/fa20-523-313/), [Edit](https://github.com/cybertraining-dsc/fa20-523-313/blob/main/project/project.md)

{{% pageinfo %}}

## Abstract

Predicting the stock market is a complex task with lots of different variables comes into play. While it is difficult to predict the short-term volatility, there are several approaches to forecast the stock market in the long term.  In this paper, we will analyze multiple methods to forecast the Indonesian Stock Exchange (IDX) recovery after the Covid-19 pandemic. The forecast will be based on technical analysis, fundamental analysis, and sentiment analysis.

Contents

{{< table_of_contents >}}

{{% /pageinfo %}}

**Keywords:** stock, market, predictive analytics, LSTM, random forest, regression, fundamental analysis, technical analysis, sentiment analysis, pandemic 


## 1. Introduction

The COVID-19 Pandemic is not just a crisis in the public health sector. It also impacts unemployment rates, business revenues, and mass psychology, which in the end lead to crashes in global stock markets. While some stock indexes like the Dow Jones Industrial Average (DJIA) and NASDAQ Composite have already recovered, the Indonesian Stock Market Index (IDX Composite) is still far below its price before the pandemic.
Some of the possible causes are: 

1. Foreign investments represent about 50% of the total fund in the IDX stock exchange. In a pandemic situation, foreign investors might choose to withdraw their stocks and find another safer country to invest in.
2. Unpredictability of the pandemic situation drives investors to reallocate their funds in safer assets, such as cash, gold, or USD.
3. Changes in the macroeconomic situation, such as unemployment rate, Indonesian Rupiah (IDR) exchange rate, and interest rate. 
4. Changes in the consumer buying power also change the business revenues, thus changing fundamental data.
5. Mass psychology of investors that the stock market is not safe in this pandemic situation, holding them from returning to the stock market

To predict the time needed for IDX Composite to recover, two indicators can be utilized:

1. Fundamental indicators, which represent the financial aspect. This can be in the form of macroeconomic data and a company financial report 
2. Technical indicators, which represent the mass psychology of investors. This can be obtained from statistical analysis of how the stock market moves
3. Sentiment analysis, which represents the mass psychology of Indonesian people. This can be obtained from Twitter and Google Trends

## 2. Background Research and Previous Work

Predicting the stock market offers great profit, thus it is a widely popular research area. While there is no specific paper addressing how to predict market recovery, there are some researches related to this area that can be utilized in this study.

### 2.1 Predicting Long Term Stock Price Movement using the Random Forest Algorithm

Milosevic [^6] uses some machine learning algorithms to predict whether some company’s value will be 10% higher or not over the period of one year. He evaluated the companies’ financial indicators and found out that the Random Forest algorithm can get 0.765 precision, which is an amazing performance in the stock market. 

### 2.2 Predicting the Next Recession using Long Short-term Memory (LSTM) Algorithm

Another long term prediction in this area is from Khedkar [^7]. He used LSTM in predicting the next recession in India. In this study, the stock closing price data is used, instead of financial data – which is technical analysis. LSTM networks are well-suited to classifying, processing, and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series. The study managed to predict the stock market crash in 2020.

### 2.3 Stock Market Prediction Using Text Mining

Text mining usually accompanies the technical analysis method. We can conduct sentiment analysis on news and social media to predict whether the stock market goes up or down. Sentiments on Twitter usually indicate market volatility and volume, while sentiments on news are a better predictor of stock movement [4].

## 3. Choice of Data-sets

In predicting the outcome, these datasets will be utilized:
Yahoo Finance (finance.yahoo.com). Yahoo Finance contains a lot of both fundamental and technical data, and they are free of charge.
Twitter (twitter.com). Sentiment analysis on Twitter will be conducted to represent the mass psychology regarding the economic condition in Indonesia

## 4. Methodology

In this study, we will predict market movement in the next two years using technical approach accompanied by sentiment analysis. While fundamental approach is a strong predictor of the long term stock price movement for an individual stock, it is very difficult to utilize it in an index like IDX composite because it will takes too many variables. The technical approach will utilize the bidirectional gated recurrent unit (BGRU), which is considered a variant of LSTM. In the second approach, we will use news from Twitter's official accounts and historical stock prices to predict the market.  

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

