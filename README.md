[<back to portfolio](https://mickael-wajnberg.github.io/)

<br>
go to [Github Repository](https://github.com/mickael-wajnberg/TimeSeriesForecasting)
<br>
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
- <br>
<img src="timeSeriesFigures/N3_16.png?raw=true"/>
<img src="timeSeriesFigures/N3_17.png?raw=true"/>
<br>
- we use Aikake Information Criterion to find the rank p,q of the ARMA(p,q) process
| Coordinates | Value      |
|-------------|------------|
| (3, 2)      | 27991.0639 |
| (2, 3)      | 27991.2875 |
| **(2, 2)** 🔴 | **27991.6036** |
| (3, 3)      | 27993.4169 |
| (1, 3)      | 28003.3496 |
| (1, 2)      | 28051.3514 |
| (3, 1)      | 28071.1555 |
| (3, 0)      | 28095.6182 |
| (2, 1)      | 28097.2508 |
| (2, 0)      | 28098.4077 |
| (1, 1)      | 28172.5100 |
| (1, 0)      | 28941.0570 |
| (0, 3)      | 31355.8021 |
| (0, 2)      | 33531.1793 |
| (0, 1)      | 39402.2695 |
| (0, 0)      | 49035.1842 |

- 
</details>
<details>
<summary>Notebook 4 : </summary>






 </details>
 
[<back to portfolio](https://mickael-wajnberg.github.io/)
