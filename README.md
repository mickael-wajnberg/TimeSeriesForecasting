[<back to portfolio](https://mickael-wajnberg.github.io/)

<br>
go to [Github Repository](https://github.com/mickael-wajnberg/TimeSeriesForecasting)
<br><br><br><br>
---
Here you will find practical work on time series forcasting in the different notebooks
<br><br>
<details>
<summary>Notebook 1 : Introduction</summary>
- predict the quarter dividend (earning per share) from johnson and johnson
<br>
<img src="timeSeriesFigures/N1_1.png?raw=true"/>
<br>
- establish seasonality
<br>
<img src="timeSeriesFigures/N1_seasonality.png?raw=true"/>
<br>
- models are historical mean, last year mean, last value, naive seasonal copy of last year
<br>
<img src="timeSeriesFigures/N1_split.png?raw=true"/>
 <img src="timeSeriesFigures/N1_mean_pred.png?raw=true"/>
 <img src="timeSeriesFigures/N1_last_pred.png?raw=true"/>
 <img src="timeSeriesFigures/N1_seasonality_pred.png?raw=true"/>
<br>
- evaluation is made by MAPE
 <br>
<img src="timeSeriesFigures/N1_results.png?raw=true"/>
<br>
</details>
<details>



 
<summary>Notebook 2 : Random Walks</summary>
- established the GOOGL stock market (google) is a random walk with Augmented Dickey-Fuller and Autocorrelation. so, it cannot be predicted by itself well.
<br>
<img src="timeSeriesFigures/N2_dataset.png?raw=true"/>
<img src="timeSeriesFigures/N2_autocorrel.png?raw=true"/>
<br>
- year ahead predictions are simply made by drift, last value and mean and evaluated through MSE
<br>
<img src="timeSeriesFigures/N2_predictions.png?raw=true"/>
<img src="timeSeriesFigures/N2_results.png?raw=true"/>
<br>
- another approch very anaive is to predict a copy of the last value
<br>
<img src="timeSeriesFigures/N2_onestep.png?raw=true"/>
<img src="timeSeriesFigures/N2_resultsF.png?raw=true"/>
<br>
</details>
<details>
<summary>Notebook 3 : Simple Statistic Modeling</summary>

 - study of the widget sales of XYZ widget company over 500 days
 <br>
<img src="timeSeriesFigures/N3_1dataset.png?raw=true"/>
<br>
 - ensure there is no seasonality 
   <br>
<img src="timeSeriesFigures/N3_2.png?raw=true"/>
<br>
 - auto-correlation is not abruptly dropping -> not a random walk -> can be predicted
   <br>
<img src="timeSeriesFigures/N3_3.png?raw=true"/>
<br>
- does auto-correlation coefficient dropping after a certain lag -> we differentiate and check autocorrelation rank 
   <br>
<img src="timeSeriesFigures/N3_4.png?raw=true"/>
<img src="timeSeriesFigures/N3_5.png?raw=true"/>
<br>
- a rank 2 is found -> it is a Moving Average (MA) rank 2 process -> we make prediction on the differentiated series after training a MA(2)
  <br>
<img src="timeSeriesFigures/N3_6.png?raw=true"/>
<img src="timeSeriesFigures/N3_7.png?raw=true"/>
<br>
- then since we found the champion model on differentiated serie, we apply it to non differentiated
   <br>
<img src="timeSeriesFigures/N3_8.png?raw=true"/>
<br><br><br>

- we work on a second dataset to predict average weekly traffic in a retail store
<br>
<img src="timeSeriesFigures/N3_9.png?raw=true"/>
<br>
- this time, even after differenciation we do not see an abrupt drop in the auto correlation -> not a moving average
 <br>
<img src="timeSeriesFigures/N3_10.png?raw=true"/>
<img src="timeSeriesFigures/N3_11.png?raw=true"/>
<br>
- partial coefficient might be in action so we plot a partial autocorrelation
 <br>
<img src="timeSeriesFigures/N3_12.png?raw=true"/>
<br>
- Since it drops, we are in an autoregressive process (order 3, since three coefficients are outside the confidence interval)
- we train a AR(3) Model and compare it to prediction using last point (our winner for GOOGL stock) and mean
<br>
<img src="timeSeriesFigures/N3_13.png?raw=true"/>
<img src="timeSeriesFigures/N3_14.png?raw=true"/>
<br>
- In a last scenario, let's explore when a dataset has both the properties MA and AR : the hourly bandwidth usage of a data center
<br>
<img src="timeSeriesFigures/N3_15.png?raw=true"/>
<br>
- We can see a slow decay autocorrelation and an alternating pattern in partial autocorrelation
<br>
<img src="timeSeriesFigures/N3_16.png?raw=true"/>
<img src="timeSeriesFigures/N3_17.png?raw=true"/>
<br>
- we use Aikake Information Criterion to find the rank p,q of the ARMA(p,q) process
<br>
<img src="timeSeriesFigures/N3_t.png?raw=true" style="width: 30%; height: auto;"/>
<br><br>
- in the top 3, the less complex model is (2,2) we evaluate the model quality by residual analysis (QQPlots, Ljung-Box tests, histogram of residual distribution, autocorrelation on residuals)
<br>
<img src="timeSeriesFigures/N3_18.png?raw=true"/>
<br>
- finally we make prediction on the differentiated model and see the ARMA model performs bettesr
<br>
<img src="timeSeriesFigures/N3_19.png?raw=true"/>
<img src="timeSeriesFigures/N3_20.png?raw=true"/>
<br>
- finally we apply the results to the original dataset
<br>
<img src="timeSeriesFigures/N3_21.png?raw=true"/>
<br>
</details>
<details>
<summary>Notebook 4 : ARMA to SARIMAX</summary>
- again let's predict the quarter dividend (earning per share) from johnson and johnson
<br>
<img src="timeSeriesFigures/N1_1.png?raw=true"/>
<br>
- we observe that by first differentiation the series is not stationary but on second differenciation it is, with Augmented Dickey-Fuller (ADF)
<br>ADF Statistic original: 2.7
<br>p-value: 1.0<br>
<br>ADF Statistic diff1: -0.4
<br>p-value diff1: 0.9<br>
<br>ADF Statistic diff2: -3.5
<br>p-value diff2: 0.006<br><br>
- by fitting an ARIMA model with I=2 and using AIC (cf notebook3) to find AR and MA are rank 3 we can make a residual evaluation to see the model residuals are effectively the unpredictable part
<br>
<img src="timeSeriesFigures/N4_1.png?raw=true"/>
<br>
- we compare ARIMA to naive seasonal (our best baseline, cf notebook1) 
<br>
<img src="timeSeriesFigures/N4_2.png?raw=true"/>
<img src="timeSeriesFigures/N4_3.png?raw=true"/>
<br>
- let's take another seasonal dataset : showing the number of passengers in a flight company per month
<br>
<img src="timeSeriesFigures/N4_4.png?raw=true"/>
<br>
- autocorrelation show clear periodic patterns
<br>
<img src="timeSeriesFigures/N4_5.png?raw=true"/>
<img src="timeSeriesFigures/N4_6.png?raw=true"/>
<br>
- patterns seems to be every 12 we can confirm visually with Fourier and spectral analysis, plotting seasonality
 <br>
<img src="timeSeriesFigures/N4_7.png?raw=true"/>
<img src="timeSeriesFigures/N4_8.png?raw=true"/>
<img src="timeSeriesFigures/N4_9.png?raw=true"/>
<br>
- it can also be confirmed with statistical test such as ADF over seasonal differenced series, a chi2 test if we bin data per 12 and kruskal wallis (KW is inconclusive here)
<br><br>'ADF Statistic': -3.383020726492479,
 <br>'p-value': 0.011551493085515039,
<br><br>Chi-Square Statistic: 292.61636904761906
<br>P-Value: 5.1233345885199216e-21
<br>Degrees of Freedom: 99
<br><br>KruskalResult(statistic=11.148400259640129, pvalue=0.430915880610989)
 <br>
 - we fit a SARIMA after a selection by AIC and evaluate the residuals 
 <br>
<img src="timeSeriesFigures/N4_11.png?raw=true"/>
<br>
 - results are compared with MAPE
  <br>
<img src="timeSeriesFigures/N4_12.png?raw=true"/>
<img src="timeSeriesFigures/N4_13.png?raw=true"/>
<br>
- now we use USA realGDP to incorporate predictions with outside values (exogenous variables) and finally complete SARIMAX 
 <br>
<img src="timeSeriesFigures/N4_14.png?raw=true"/>
<br>
- with same process of AIC + residuals we find 
<br>
<img src="timeSeriesFigures/N4_15.png?raw=true"/>
<br>
it doesn't look like much but the difference is in M$
 
</details>


<details>
<summary>Notebook 5 : Mutliple Intricated Time Series</summary>
 - we consider here the case where two times series affect each others : real disposable income and real conumption in USA
<br>
<img src="timeSeriesFigures/N5_1.png?raw=true"/>
<br>
- we use the VARMAX model and we consider AIC to find the best rank : best found = 3
- we now use granger causality tests to determine if one series causes the other 
 realcons Granger-causes realdpi?

<br>------------------
<br>
<br>Granger Causality
<br>number of lags (no zero) 3
<br>ssr based F test:         F=9.2363  , p=0.0000  , df_denom=192, df_num=3
<br>ssr based chi2 test:   chi2=28.7191 , p=0.0000  , df=3
<br>likelihood ratio test: chi2=26.8268 , p=0.0000  , df=3
<br>parameter F test:         F=9.2363  , p=0.0000  , df_denom=192, df_num=3
<br>
<br>realdpi Granger-causes realcons?
<br>
<br>------------------
<br>
<br>Granger Causality
<br>number of lags (no zero) 3
<br>ssr based F test:         F=2.8181  , p=0.0403  , df_denom=192, df_num=3
<br>ssr based chi2 test:   chi2=8.7625  , p=0.0326  , df=3
<br>likelihood ratio test: chi2=8.5751  , p=0.0355  , df=3
<br>parameter F test:         F=2.8181  , p=0.0403  , df_denom=192, df_num=3
<br>
- granger causality exists both ways so there is some correlation effect
- after checking residuals with the selected model (random in both series) we evaluatepredictive model
    <br>
<img src="timeSeriesFigures/N5_3.png?raw=true"/>
<img src="timeSeriesFigures/N5_2.png?raw=true"/>
<br>
 </details>


<details>
<summary>Notebook 6 : Preparing framework for Deep Learning </summary>
- we have a large dataset of hourly traffic volume in interstate metro but data is not "clean" , we also have multiple variables at our disposal but uncomplete (display here the number of missing values)
 <br>
<img src="timeSeriesFigures/N6_1.png?raw=true"/>
 <img src="timeSeriesFigures/N6_2.png?raw=true"/>
<br>
- we drop some columns and fill the empty values per median  
<br>
<img src="timeSeriesFigures/N6_3.png?raw=true"/>
<br>
- since data is too dense for eye sight, we zoom in to see that there is daily weekly seasonality in traffic volume but also in temperature
 <br>
<img src="timeSeriesFigures/N6_4.png?raw=true"/>
 <img src="timeSeriesFigures/N6_5.png?raw=true"/>
<br>
- to address seasonality we do feature engineering creating cos and sin on hour time and adding day name and month as separate features
- we define classes for visalizing results and define baseline models on sliding windows
- baseline 1 is a 1 time stamp shift (predicting each values is like the previous one) 
 <br>
<img src="timeSeriesFigures/N6_6.png?raw=true"/>
<br>
 - baseline 2 is last value of the input part  replicated on as many prediction steps desired (by default as much as input)
 <br>
<img src="timeSeriesFigures/N6_7.png?raw=true"/>
<br>
 - baseline 3 is copying exactly the input in the output 
 <br>
<img src="timeSeriesFigures/N6_8.png?raw=true"/>
<br>
 - baseline 4 is copying the input average in the output as manytime as desired
 <br>
<img src="timeSeriesFigures/N6_9.png?raw=true"/>
<br>
 - we can then evaluate the results and be ready to compare our next evolved deep learning models of notebooks 7 and 8 
  <br>
<img src="timeSeriesFigures/N6_10.png?raw=true"/>
<br>
  </details>


<details>
<summary>Notebook 7 : Deep Learning basic architectures</summary>
- in previous notebook we developped baseline here we create different models the first one being a simple regression using only traffic volume
- there are three scenarii : predicting the next value on one variable, predicting few next values on one variable and predicting next few values on multiple variables.
- we try to predict a single value with the linear regression 
 <br>
<img src="timeSeriesFigures/K7_1.png?raw=true"/>
<br>
 - then we extend to a full day of prediction 
 <br>
<img src="timeSeriesFigures/K7_2.png?raw=true"/>
<img src="timeSeriesFigures/K7_3.png?raw=true"/>
<br>
 - we can also easily predict the output for two variables : traffic volume and temperature
 <br>
<img src="timeSeriesFigures/K7_4.png?raw=true"/>
<img src="timeSeriesFigures/K7_5.png?raw=true"/>
<img src="timeSeriesFigures/K7_7.png?raw=true"/>
<br>
 - Now that we have baselines with linear regressions we can go to deep learning models, adding two layers of ReLU (64 units) 
  <br>
<img src="timeSeriesFigures/K7_7.png?raw=true"/>
<img src="timeSeriesFigures/K7_8.png?raw=true"/>
<br>
   </details>


<details>
<summary>Notebook 8 : Time Series Deep Learning Specific Architectures</summary>
- First architecture that we can use for time series analysis is LSTM<br>
- We observe that for multistep - one variable LSTM produces a better result than linear regression and simple deep neural network <br>
   <br>
<img src="timeSeriesFigures/A8_1.png?raw=true"/>
<br>
- we tried both CNN and CNN+LSTM architectures but it still seems LSTM out perform them <br>
   <br>
<img src="timeSeriesFigures/A8_2.png?raw=true"/>
<br>
</details>
  
[<back to portfolio](https://mickael-wajnberg.github.io/)
