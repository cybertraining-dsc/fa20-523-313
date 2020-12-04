# Analyzing LSTM Performance on Predicting Stock Market for Multiple Time Steps

[![Check Report](https://github.com/cybertraining-dsc/fa20-523-313/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-313/actions)
[![Status](https://github.com/cybertraining-dsc/fa20-523-313/workflows/Status/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-313/actions)
Status: in progress



Fauzan Isnaini, [fa20-523-313](https://github.com/cybertraining-dsc/fa20-523-313/), [Edit](https://github.com/cybertraining-dsc/fa20-523-313/blob/main/project/project.md)

{{% pageinfo %}}

## Abstract

Predicting stock market has been an attractive field of research for a long time because it promises big wealth for anyone who can find the secret sauce. For a long time, traders around the world have been relying on technical analysis to analyze patterns in the stock price movement and predict the trend. With the advancement of big data, some financial institutions are beginning to predict the market by creating a model of the market using machine learning. While some researches produce promising results, most of them are directed on predicting the next day market behavior. In this study, we created an LSTM model to predict the market for multiple time steps. We then analyzed the performance of the model for different time period. From our observations, LSTM is good at predicting 5 time steps ahead, but the prediction became inaccurate as the time frame gets longer. 



Contents

{{< table_of_contents >}}

{{% /pageinfo %}}

**Keywords:** stock, market, predictive analytics, LSTM, random forest, regression, technical analysis 


## 1. Introduction

Predicting the stock market is a complex task with lots of different variables comes into play. As Giles [^1] explained, financial forecasting is an instance of signal processing problem which is difficult because of high noise, small sample size, non-stationary, and non-linearity. 

The noisy characteristics mean the incomplete information gap between past stock trading price and volume with a future price. The stock market is sensitive with the political and macroeconomic environment. However, these two kinds of information are too complex and unstable to gather. The above information that cannot be included in features are considered as noise. 

The sample size of financial data is determined by the real world transaction records. On one hand, a larger sample size refers to a longer period of transaction records; on the other hand, large sample size increases the uncertainty of financial environment.

By non-stationarity, one means that the distribution of stock data is various during time changing. Non-linearity implies that feature correlation of different individual stocks is various.

Efficient Market Hypothesis was developed by Burton G. Malkiel in 1991. In Burton’s hypothesis, he indicates that predicting or forecasting the financial market is unrealistic, because price changes in the real world are unpredictable. All the changes in prices of the financial market are based on immediate economic events or news. Investors are profit-oriented, their buying or selling decisions are made according to most recent events regardless past analysis or plans. The argument about this Efficient Market Hypothesis has never been ended. So far, there is no strong proof that can verify if the efficient market hypothesis is proper or not.

However, as Yaser claims, financial markets are predictable to a certain extent. The past experience of many price changes over a certain period of time in the financial market and the undiscounted serial correlations among vital economic events affecting the future financial market are two main pieces of evidence opposing the Efficient Market Hypothesis. 

In recent years, machine learning methods have been extensively researched for their potentials in forecasting and prediction of the financial market. Multi-layer feed forward neural networks, SVM, reinforcement learning, relevance vector machines, and recurrent neural networks are the hottest topics of many approaches in financial market prediction field. Among all the machine learning methods, neural networks are well studied and have been successfully used for forecasting and modeling financial market. 

Unlike traditional machine learning models, the network learns from the examples by constructing an input-output mapping for the problem at hand. Such an approach brings to mind the study of  nonparametric statistical inference; the term nonparametric is used here to signify the fact that no prior assumptions are made on a statistical model for the input data. 

As Francis E.H. Tay and Lijuan Cao explained in their studies, Neural networks are more noise tolerant and more flexible compared with traditional statistical models. By noise tolerance, one means neural networks have the ability to be trained by incomplete and overlapped data. Flexibility refers to that neural networks have the capability to learn dynamic systems through a retraining process using new data patterns.

Long short-term memory is a recurrent neural network introduced by Sepp Hochreite and Jurgen Schmidhuber in 1997. LSTM is designed to forecast, predict and classify time series data even long time lags between vital events happened before. LSTMs have been applied to solve several of problems; among those, handwriting Recognition and speech recognition made LSTM famous. 

LSTM has advantages compared with traditional back-propagation neural networks and normal recurrent neural networks. The constant error back propagation inside memory blocks enables in LSTM ability to overcome long time lags in case of problems similar to those discussed above; LSTM can handle noise, distributed representations, and continuous values; LSTM requires no need for parameter fine-tuning, it works well over a broad range of parameters such as learning rate, input gate bias, and output gate bias.

In this study, we created an LSTM model to predict the market for multiple time steps. We then analyzed the performance of the model for different time period.

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

