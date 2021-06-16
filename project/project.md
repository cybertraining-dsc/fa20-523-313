---
date: 2021-03-15
title:  Analyzing LSTM Performance on Predicting the Stock Market for Multiple Time Steps
linkTitle: Stock MArket
tags: ["project", "ai", "finance"]
description: Analyzing LSTM Performance on Predicting the Stock Market for Multiple Time Steps
author: Fauzan Isnaini
resources:
- src: "**.{png,jpg}"
  title: "Image #:counter"
---

[![Check Report](https://github.com/cybertraining-dsc/fa20-523-313/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-313/actions)
[![Status](https://github.com/cybertraining-dsc/fa20-523-313/workflows/Status/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-313/actions)
Status: final Type: Project



Fauzan Isnaini, [fa20-523-313](https://github.com/cybertraining-dsc/fa20-523-313/), [Edit](https://github.com/cybertraining-dsc/fa20-523-313/blob/main/project/project.md)

{{% pageinfo %}}

## Abstract

Predicting the stock market has been an attractive field of research for a long time because it promises big wealth for anyone who can find the secret. For a long time, traders around the world have been relying on fundamental analysis and technical analysis to predict the market. Now with the advancement of big data, some financial institutions are beginning to predict the market by creating a model of the market using machine learning. While some researches produce promising results, most of them are directed at predicting the next day's market behavior. In this study, we created an LSTM model to predict the market for multiple time frames. We then analyzed the performance of the model for some different time periods. From our observations, LSTM is good at predicting 30 time steps ahead, but the RMSE became larger as the time frame gets longer.



Contents

{{< table_of_contents >}}

{{% /pageinfo %}}

**Keywords:** stock, market, predictive analytics, LSTM, random forest, regression, technical analysis 


## 1. Introduction

Stock market prediction is a fascinating field of study for many analysts and researchers becauseof the significant amount of money circulating in the market. While there are numerous studies conducted in this field, predicting the stock market remains a challenging task, because of its noisy and non-stationary nature [^3]. The stock market is "noisy" because it is sensitive to mass psychology. The trends and patterns in the stock market can also change abruptly because of bad news, natural disasters, and some unforeseen circumstances, thus it is considered non-stationary.

The efficient market hypothesis even suggests that predicting or forecasting the financial market is unrealistic because price changes in the real world are unpredictable. All the changes in prices of the financial market are based on immediate economic events or news. Investors are profit-oriented, their buying or selling decisions are made according to the most recent events regardless of past analysis or plans. The argument about this Efficient Market Hypothesis has never been ended. So far, there is no strong proof that can verify if the efficient market hypothesis is proper or not [^5].

However, as Mostafa [^1] claims, financial markets are predictable to a certain extent. The past experience of many price changes over a certain period of time in the financial market and the undiscounted serial correlations among vital economic events affecting the future financial market are two main pieces of evidence opposing the Efficient Market Hypothesis.

The most popular methods in predicting the stock markets are technical and fundamental analysis. Fundamental analysis is mainly based on three essential aspects [^10]: (i) macroeconomic analysis such as Gross Domestic Products and Consumer Price Index (CPI) which analyses the effect of the macroeconomic environment on the future profit of a company, (ii) industry analysis which estimates the value of the company based on industry status and prospect, and (iii) company analysis which analyses the current operation and financial status of a company to evaluate its internal value.

On the other hand, technical analysis is grouped into eight domains [^10]: sentiment, flow-of-funds, raw data, trend, momentum, volume, cycle, and volatility. Sentiment represents the behaviors of various market participants. Flow-of-funds is a type of indicator used to investigate the financial status of various investors to pre-evaluate their strength in terms of buying and selling stocks, then, corresponding strategies, such as short squeeze can be adopted. Raw data include stock price series and price patterns such as K-line diagrams and bar charts. Trend and momentum are examples of price-based indicators, trend is used for tracing the stock price trends while momentum is used to evaluate the velocity of the price change and judge whether a trend reversal in stock price is about to occur. Volume is an indicator that reflects the enthusiasm of both buyers and sellers for investing, it is also a basis for predicting stock price movements. The cycle is based on the theory that stock prices vary periodically in the form of a long cycle of more than 10 years containing short cycles of a few days or weeks. Finally, volatility is often used to investigate the fluctuation range of stock prices and to evaluate risk and identify the level of support and resistance.

While those two are still the most popular approaches, the age of big data has brought a new method to predict the stock market: quantitative analysis. In this new method, the stock market is captured into a mathematical model, and machine learning is used to predict its behavior. Research by Alzazah and Cheng [^5] analyzed more than 50 articles to compare various machine learning (ML) and deep learning (DL) methods used to find which method could be more effective in prediction and for which types and amount of data. This research has proven that quantitative analysis with LSTM gives a promising result as the predictor of a stock market.

In this study, we analyzed the performance of LSTM in predicting the stock market for multiple time frames. LSTM model is used because it is designed to forecast, predict, and classify time series data [^5]. Despite the promising result in LSTM, most of the previous studies are conducted in building a model to predict the next day's price. Thus, we wanted to know how accurate the LSTM model in predicting the stock market for a longer time frame (i.e. from daily to monthly time frame). We also chose to incorporate technical analysis rather than fundamental analysis in our model, because while fundamental analysis tends to be accurate in the yearly period, it could not predict the fluctuation in the given time frame.

## 2. Background Research and Previous Work

### 2.1 MACD in Technical Analysis

MACD [^6] is an acronym for moving average convergence/divergence. It is a widely used technical indicator to confirm either the bullish or bearish phase of the market. In essence, the MACD indicator shows the perceived strength of a downward or upward movement in price. Technically, it’s an oscillator, which is a term used for indicators that fluctuate between two extreme values, for example, from 0 to 100. 
MACD evolved from the exponential moving average (EMA), which was proposed by Gerald Appel in the 1970s. The standard MACD is the 12-day EMA subtracted by the 26-day EMA, which is also called the DIF. The MACD histogram, which was developed by T. Aspray in 1986, measures the signed distance between the MACD and its signal line calculated using the 9-day EMA of the MACD, which is called the DEA. Similar to the MACD, the MACD histogram is an oscillator that fluctuates above and below the zero line. The construction formula of MACD is given in figure 1.

![MACD Formula](https://github.com/cybertraining-dsc/fa20-523-313/raw/main/project/images/MACDFormula.png)

**Figure 1:** MACD formula [^6]

The number of the MACD histogram is usually called the MACD bar or OSC. The analysis process of the cross and deviation strategy of DIF and DEA includes the following three steps: (i) Calculate the values of DIF and DEA, (ii)When DIF and DEA are positive, the MACD line cuts the signal line in the uptrend, and the divergence is positive, there is a buy signal confirmation, and (iii)When DIF and DEA are negative, the signal line cuts the MACD line in the downtrend, and the divergence is negative, there is a sell signal confirmation.

### 2.2 Time Series Forecasting

Time series analysis and dynamic modeling [^8] is an interesting research area with a great number of applications in business, economics, ﬁnance, and computer science. The aim of time series analysis is to study the path observations of time series and build a model to describe the structure of data and then predict the future values of time series. Due to the importance of time series forecasting in many branches of applied sciences, it is essential to build an effective model with the aim of improving the forecasting accuracy. A variety of time series forecasting models have been evolved in the literature.

Time series forecasting is traditionally performed in econometric using ARIMA models, which is generalized by Box and Jenkins. ARIMA has been a standard method for time series forecasting for a long time. Even though ARIMA models are very prevalent in modeling economical and ﬁnancial time series, they have some major limitations. For instance, in a simple ARIMA model, it is hard to model the non-linear relationships between variables. Furthermore, it is assumed that there is a constant standard deviation in errors in ARIMA model, which is in practice may not be satisﬁed. When an ARIMA model is integrated with a Generalized Auto-regressive Conditional Heteroskedasticity(GARCH) model, this assumption can be relaxed. On the other hand, the optimization of a GARCH model and its parameters might be challenging and problematic. There are several other applications of ARIMA for modeling short and long-run effects of economics parameters.

Recently, new techniques in deep learning have been developed to address the challenges related to the forecasting models. LSTM (Long Short-Term Memory) is a special case of the Recurrent Neural Network (RNN) method that was initially introduced by Hochreiter and Schmidhuber. Even though it is a relatively new approach to address prediction problems. Deep learning-based approaches have gained popularity among researchers. 

LSTM is designed to forecast, predict, and classify time series data even long time lags between vital events that happened before. LSTMs have been applied to solve several problems; among those, handwriting Recognition and speech recognition made LSTM famous. LSTM has copious advantages compared with traditional back-propagation neural networks and normal recurrent neural networks. The constant error backpropagation inside memory blocks enables LSTM ability to overcome long time lags in case of problems similar to those discussed above; LSTM can handle noise, distributed representations, and continuous values; LSTM requires no need for parameter fine-tuning, it works well over a broad range of parameters such as learning rate, input gate bias, and output gate bias [^5].

### 2.3 Using LSTM in Stock Prediction and Quantitative Trading

During the pre-deep learning era, Financial Time Series modeling has mainly concentrated in the field of ARIMA and any modifications on this, and the result has proved that the traditional time series model does provide decent predictive power to a limit. More recently, deep learning methods have demonstrated better performances thanks to improved computational power and the ability to learn non-linear relationships enclosed in various financial features. 

The direction of the financial market is always stochastic and volatile and the return of the security return is deemed to be unpredictable. Analysts now are trying to apply the modeling techniques from Natural Language Processing into the field of Finance as the similarity of having the sequential property in the data. Zhou [^11] has constructed and applied the Long Short Term Memory Model (LSTM) and the traditional ARIMA model, into the prediction of stock prices on the next day. It was proven that the LSTM model performed better than the ARIMA model.

## 3. Choice of Data-sets

This project used the historical data of the Jakarta Composite Index (JKSE) from Yahoo Finance [^2]. The JKSE is a national stock index of Indonesia, which consists of 700 companies. We choose to incorporate the composite index because it has a beta value of 1, which means it has neutral volatility compared to an individual stock to be incorporated into a model. The dataset contains the Open, High, Low, Close, and Volume data for daily time period on the stock index. The daily data is taken from January 1st, 2013 until November 17th, 2020. We choose the daily data over the monthly data because it offers a more complete pattern. Figure 2 and 3 provides a snapshot of the first few rows of the daily and monthly data respectively.

![Head of Daily Data](https://github.com/cybertraining-dsc/fa20-523-313/raw/main/project/images/newdailyhead.png)


**Figure 2:** Snapshot of the first rows of the daily data

![Head of Monthly Data](https://github.com/cybertraining-dsc/fa20-523-313/raw/main/project/images/newMonthlyHead.png)

**Figure 3:** Snapshot of the first rows of the monthly data

We also used the MACD technical indicator as an input to our model. The MACD parameters are generated using the ta-lib library [^7] based on the Yahoo Finance data. Figure 4 and 5 provides a snapshot of the first few rows of the daily and monthly data respectively after incorporating the MACD technical indicator.

![Daily MACD](https://github.com/cybertraining-dsc/fa20-523-313/raw/main/project/images/MACDonDaily.png)

**Figure 4:** MACD on the daily data

![Monthly MACD](https://github.com/cybertraining-dsc/fa20-523-313/raw/main/project/images/newMACDonMonthly.png)

**Figure 5:** MACD on the monthly data

## 4. Methodology

### 4.1 Technology

Python [^4] was the language of choice for this project. This was an easy decision for these reasons [^7]: 

1. Python as a language has an enormous community behind it. Any problems that might be encountered can be easily solved with a trip to Stack Overflow. Python is among the most popular languages on the site which makes it very likely there will be a direct answer to any query.
2. Python has an abundance of powerful tools ready for scientific computing. Packages such as Numpy, Pandas, and SciPy are freely available and well documented. Packages such as these can dramatically reduce, and simplify the code needed to write a given program. This makes iteration quick.
3. Python as a language is forgiving and allows for programs that look like pseudo code. This is useful when pseudocode given in academic papers needs to be implemented and tested. Using Python, this step is usually reasonably trivial.

In building the LSTM model, Keras [^9] library is used. It contains numerous implementations of commonly used neural network building blocks such as layers, objectives, activation functions, optimizers, and a host of tools to make working with image and text data easier. The code is hosted on GitHub, and community support forums include the GitHub issues page, a Gitter channel, and a Slack channel.

### 4.2 Data Preprocessing

After downloading the historical datasets from Yahoo Finance, the MACD technical indicator is generated using the ta-lib library. Because MACD needs to capture data from the previous time period, the MACD values on the first rows of the data are missing. These rows are then removed before being split into 8:2 proportions for training and testing purposes in the LSTM model.

### 4.3 The LSTM Model

A multivariate LSTM model with two hidden layers is used, with a dropout parameter of 0.2. Adam is used as the optimization algorithm.  The model uses 90 days time steps, which means it uses the past 90 days of data to predict the output. It has 8 features, which are the Close, Low, High, Open, Volume, MACD, MACD Signal, and MACD Histogram. It then gives one output, which is the open price for the given time frame. We then analyze the performance of our model for each of the time frames.

## 5. Results

We used callback function to find the best number of epochs in the model. Figure 6 gives the mean squared error (MSE) curve of the prediction in the training dataset for each given epoch. It shows that the MSE converges after 20 epochs, with a value of 0.0243.

![Epoch Loss](https://github.com/cybertraining-dsc/fa20-523-313/raw/main/project/images/newlossepochs.png)

**Figure 6:** MSE on the training data for each given epoch 

Figure 7 shows the root mean squared error (RMSE) on the testing dataset for each time frame. It clearly shows that the RMSE becomes bigger on a longer time frame. When predicting the next day period, the RMSE is 323.41, while when predicting 30 days ahead, the RMSE increase to 481.32. But overall, these values are still acceptable because they are smaller than the standard deviation of the actual dataset of 667.31.

![RMSE](https://github.com/cybertraining-dsc/fa20-523-313/raw/main/project/images/RMSEonTimeFrame.png)

**Figure 7:** RMSE on the training data for each time frame 

Figure 8 and Figure 9 compare the predicted values on the training data for 1 day and 30 days time frames respectively, while Figure 10 and 11 give the comparison on the test data. It can be seen that the model cannot predict steep ramps in the price change, thus it is lagged from the actual price. The predicted price becomes further lagged when predicting for a longer time frame, thus resulting in a bigger RMSE.

![Next Day Prediction](https://github.com/cybertraining-dsc/fa20-523-313/raw/main/project/images/newOneDayPredict.png)

**Figure 8:** Comparison between the next day prediction and its actual values based on the training data

![30 Days Prediction](https://github.com/cybertraining-dsc/fa20-523-313/raw/main/project/images/newThirtyDaysPredict.png)

**Figure 9:** Comparison between the 30 days time frame prediction and its actual values based on the training data

![Next Day Test](https://github.com/cybertraining-dsc/fa20-523-313/raw/main/project/images/newonedaytest.png)

**Figure 10:** Comparison between the next day prediction and its actual values based on the testing data

![30 Days Test](https://github.com/cybertraining-dsc/fa20-523-313/raw/main/project/images/thirtydaystest.png)

**Figure 11:** Comparison between the 30 days time frame prediction and its actual values based on the testing data

We also found that a longer training dataset does not always give a better prediction because the model might overfit with the training data. In fact, when we used historical market data from January 2000, the RMSE became close to 3,000. This might be due to overall the stock market tends to always get higher every year. When the data is too old, the model needs to compensate for the time when the price is still very low.

We also capture the time needed to run each critical process using cloudmesh-common benchmark and stopwatch framework [^6]. The stopwatch recordings are shown in Table 1. The table shows that training the model took the longest time. It also highlights the system's specification used in running the program.

**Table 1:** Benchmark results

![Table 1](https://github.com/cybertraining-dsc/fa20-523-313/raw/main/project/images/benchmark1.png)

![Stopwatch](https://github.com/cybertraining-dsc/fa20-523-313/raw/main/project/images/stopwatch.png)

## 6. Conclusion and Future Works

We have analyzed the performance of LSTM in predicting the stock price for different time frames. While it gives a promising result in predicting the next day's price, the prediction becomes less accurate for a longer time frame. This might be due to the non-stationarity nature of the stock market. The stock market trends can change abruptly because of a sudden change in the political and economic conditions. Using the daily market data, our model gives promising results within 30 days time frame.
This project has analyzed the performance of LSTM using RMSE, but further research may measure the performance based on the potential financial gain. After all, the stock market is a place to make money, thus financial gain is a better metric of performance.
Further improvement may also be done on our model. We only used price data and MACD technical indicator for the prediction. Further research may utilize other technical indicators, such as RSI and Stochastics to get a better prediction.

## 7. Acknowledgements

The author would like to thank Dr. Geoffrey Fox, Dr. Gregor von Laszewski, and the associate instructors in the FA20-BL-ENGR-E534-11530: Big Data Applications course (offered in the Fall 2020 semester at Indiana University, Bloomington) for their continued assistance and suggestions concerning exploring this idea and also for their aid with preparing the various drafts of this article.

## 8. References

[^1]: A. Mostafa and Y. S., "Introduction to financial forecasting. Applied Intelligence," Applied Intelligence, 1996. 

[^2]: "Composite Index (JKSE) Charts, Data &amp; News," Yahoo! Finance, 08-Dec-2020. [Online]. Available: <https://finance.yahoo.com/quote/^JKSE/>. [Accessed: 08-Dec-2020]. 

[^3]: D. Shah, H. Isah, and F. Zulkernine, "Stock Market Analysis: A Review and Taxonomy of Prediction Techniques," International Journal of Financial Studies, vol. 7, no. 2, p. 26, 2019. 

[^4]: F. Isnaini, "cybertraining-dsc/fa20-523-313," GitHub, 08-Dec-2020. [Online]. Available: <https://github.com/cybertraining-dsc/fa20-523-313/blob/main/project/code/multivariate.ipynb>. [Accessed: 08-Dec-2020].

[^5]: F. S. Alzazah and X. Cheng, "Recent Advances in Stock Market Prediction Using Text Mining: A Survey," E-Business [Working Title], 2020.

[^6]: G. von Laszewski, "cloudmesh/cloudmesh-common," GitHub, 2020. [Online]. Available: <https://github.com/cloudmesh/cloudmesh-common>. [Accessed: 08-Dec-2020]. 

[^7]: J. Bosco and F. Khan, Stock Market Prediction and Efficiency Analysis using Recurrent Neural Network. Berlin, Germany: 2018, 2018.

[^8]: J. Wang and J. Kim, "Predicting Stock Price Trend Using MACD Optimized by Historical Volatility."

[^9]: TA-Lib. [Online]. Available: <https://mrjbq7.github.io/ta-lib/func_groups/momentum_indicators.html>. [Accessed: 08-Dec-2020].

[^10]: S. Siami-Namini, N. Tavakoli, and A. S. Namin, "A Comparison of ARIMA and LSTM in Forecasting Time Series," 2018 17th IEEE International Conference on Machine Learning and Applications (ICMLA), 2018. 

[^11]: V. Bielinskas, "Multivariate Time Series Prediction with LSTM and Multiple features (Predict Google Stock Price)," Youtube, 2020. [Online]. Available: <https://www.youtube.com/watch?v=gSYiKKoREFI>. [Accessed: 08-Dec-2020].

[^12]: Y. Hu and X. Zhang, "Application of evolutionary computation for rule discovery in stock algorithmic trading," Applied Soft Computing, 2015.

[^13]: Z. Zhou, "Using LSTM in Stock prediction and Quantitative Trading," CS230, 2020.
