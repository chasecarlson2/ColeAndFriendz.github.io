# [ColeAndFriendz](https://chasecarlson2.github.io/ColeAndFriendz.github.io/) Team Project Website

Hi, welcome to ColeAndFriendz website! The team consists of Cole Humes, Chase Carlson, Priya Bhatnagar, and Jake Betlow. This is a website to showcase our final project for FIN 377 - Data Science for Finance course at Lehigh University.

To see the complete analysis file click [here](https://github.com/chasecarlson2/ColeAndFriendz.github.io/blob/8281b73a895d26380c0be030efd4ac2b9895a312/notebooks/model_analysis.ipynb).

## **Table of contents**
1. [Introduction](#introduction)
2. [Methodology](#meth)
3. [Analysis Section](#section3)
4. [Summary](#summary)

## **Introduction**  <a name="introduction"></a>

Predicting stock prices accurately using machine learning is a very difficult task, but if done with some accuracy can be very financially rewarding. The main goal of this project is to use the comprehensive CPI dataset to predict monthly stock prices for the S&P 500 index as well as for specific sectors within the index. Using measurements such as monthy % changes, real risk premium, and inflation risk premium alongside inflation predictor variables, we aim to reach an R^2 value within .01-.02 or greater. By aiming for an R^2 we do realize we are giving ourselves some room for error.

## **Methodology** <a name="meth"></a>

We know that stock prices are affected by more than inflation rates. However, if sector related inflation rates are expected to rise, the price of goods and services are expected to also rise and investors are more likely to to invest in stocks which will inherently drive up the price of stocks. 

By looking at inflation trends and then using our model to predict inflation numbers for individual sectors,  we can come up with an idea about the actions investors may take in the future which will affect stock prices. 

Here is the code for some of the graphs below:

```python
sns.barplot(x="energy_inflation", y="Date", data=df2).set(title='Energy Inflation by Date')
inflation_map = df2.pivot("Month", "Year", "actual_inflation")
ax = sns.heatmap(inflation_map).set(title='Inflation by Date')
plt.rcParams['figure.figsize'] = [15, 15] 
plt.show()

fig = plt.figure(figsize=(20, 5))
fig1 = fig.add_subplot(121); sns.scatterplot(y=df2.spy_price, x=df2['Real Risk Premium'], palette='YlOrRd')
fig2 = fig.add_subplot(122); sns.regplot(x='Inflation Risk Premium', y='spy_price', data=df2)
fig1.title.set_text('Price vs Real Risk Premium')
fig2.title.set_text('Price vs Inflation Risk Premium')
plt.rcParams['figure.figsize'] = [15, 15] 
plt.show()

figB = plt.figure(figsize=(20, 5))
fig4 = figB.add_subplot(122); sns.regplot(x='Expected Inflation', y='energy_price', data=df2)
fig4.title.set_text('Energy Price vs Expected Inflation')
plt.rcParams['figure.figsize'] = [15, 15] 
plt.show()
```


Here are the graphs developed from our dataset representing historical inflation by date and sector price vs. expected inflation correlation. 

![](pics/Screen Shot 2022-05-02 at 4.19.24 PM.png)
<br><br>
Analysis: In this heat map we can see a few things. First, during the financial crisis we see significantly different negative values shown by the darkest part of the graph. Second, 2015 has a dark portion reflecting trends noted in the previous graph. Lastly, from 2021 onwards we can see a trend of slowly elevating inflation beginning.

![](pics/Screen Shot 2022-05-02 at 4.01.40 PM.png)
<br><br>
Analysis: This graph shows that there is not a correlation in every relationship. The inflation risk premium is not correlated with spy prices.

![](pics/Screen Shot 2022-05-02 at 4.02.02 PM.png)
<br><br>
Analysis: In this graph we see that there is a relationship between expected inflation and prices in the health sector. It is not a particularly strong relationship, when expected inflation is above approximately 2.25, changes in health price are minimal. However, there is increased variability in health prices below the 2.25 mark. Inflation expectations are not the only determinant of health price, so this relationship might not hold true in the future.

![](pics/Screen Shot 2022-05-02 at 4.19.01 PM.png)
<br><br>
Analysis: When performing a similar analysis on the energy sector, inflation expectations have a different historical pattern. The point at which variability in energy prices is minimal is at 2.75 and above. Additionally, there is an even greater variability in these prices compared to the health sector. There appears to be a weak trend line for these values, but not one that alone could explain a large variance in the data.

## **Analysis** <a name="section3"></a>

Below is the code used to create our health sector predictive model. We used similar code for the real estate and energy sectors also.

```python
#Health Predictive Model 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df2 = pd.read_csv("final_df.csv")
y = df2.health_price
y = y.iloc[1:]
housing = df2.drop('health_price', axis=1)
housing = housing[:-1]

rng = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(housing, y, random_state=rng, train_size=0.8)

numeric_cols = ['expected_health', #'expected_real_estate', #'expected_energy',#'RE_inflat_ratio',# 'energy_inflat_ratio',
                #'actual_inflation',# 'heath_inflat_ratio', #'real_estate_price', 'health_price', 'energy_price',
                #'real_estate_inflation', #'health_inflation', 'energy_inflation', 'Expected Inflation',
                'Real Risk Premium','Inflation Risk Premium', #'Expected Inflation'
               ]
numeric_pipe = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
preproc_pipe = ColumnTransformer(
    [  # arg 1 of ColumnTransformer is a list, so this starts the list
        # a tuple for the numerical vars: name, pipe, which vars to apply to
        ("num_impute", numeric_pipe, numeric_cols),
        # a tuple for the categorical vars: name, pipe, which vars to apply to
    ]
    ,  # ColumnTransformer can take other args, most important: "remainder"
    remainder='drop'  # you either drop or passthrough any vars not modified above
)
pipe = make_pipeline(preproc_pipe, linear_model.ElasticNet(random_state=rng))
for i in pipe.get_params():
    print(i)
param_1_List = [0.0150, 0.0151, 0.0152, 0.0153, 0.0154, 0.0155, 0.0156, 0.0157]
param_2_List = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
parameters = {'elasticnet__alpha': param_1_List, 'elasticnet__l1_ratio': param_2_List}
grid_search = GridSearchCV(estimator=pipe,
                           param_grid=parameters,

                 )
grid_search.fit(X_train, y_train)
print("The optimized parameters output of the grid search are:")
print(grid_search.best_params_)
print("The score of the optimized output of the grid search on the training data is: " + str(grid_search.best_score_))
scores = grid_search.score(X_test, y_test)
print("Using our optimized model, the R2 score on the hold out data is: " + str(scores))
predictions = grid_search.best_estimator_.predict(X_test)
index = 0

predictionDF = pd.DataFrame(predictions)
predictionDF.describe()
y_test_val = y_test[0:]
y_testDF = pd.DataFrame(y_test_val)
d = {'Prediction':predictions,'Actual':y_test[0:]}
Health_pred = pd.DataFrame(d, columns=['Prediction','Actual'])
Health_pred
```
Health Sector Model R2: **0.005222862**
<br><br>
Real Estate Sector Model R2: **0.043317521**
<br><br>
Energy Sector Model R2: **0.012843825**

## **Summary** <a name="summary"></a>

Blah blah



## **About the team**

<img src="..." alt="Chase" width="300"/>
<br>
Chase is a senior at Lehigh studying finance.
<br><br><br>
<img src="..." alt="Cole" width="300"/>
<br>
Cole is a senior at Lehigh studying finance.


## More 

To view the GitHub repo for this website, click [here](https://github.com/donbowen/teamproject).
