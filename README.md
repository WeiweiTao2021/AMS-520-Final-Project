# AMS-520-Final-Project

In this project, we are aiming to build classification models to predict the dynamics of mid-price and price spread of limit order book. The data we used is the high-frequency tranding and quote data for APPL on Jan 06. 

Our code contains following parts:
1. Data cleaning implemented in taq_data_cleaning.py: removed the first and last 15 minutes during usual trading hours; defining MOX id to filter quote events happening at the same timestamp. 

2. Data modeling implemented in (Predict Price Movement XGB_3class.ipynb, Predict Price Movement XGB_Binary Classification): performed train/test splitting, data resampling and build XG boost model for classificaiton.

In addition to XGB, we have also implemented the Random Forest model with spline transformation. Results indicate that RF generated better performance as comparing to XGB in our project.

## Referecen:

Alec N.Kercheval, Yuan Zhang, 2013, Modeling high-frequency limit order book dynamics with support vector machines
