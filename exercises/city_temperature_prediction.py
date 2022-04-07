from http.client import responses
from statistics import mean
import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename , parse_dates =['Date'])
    df.dropna(inplace=True)
    df = df[df['Temp']>=-10]
    responses = df['Temp']
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df.drop(['Temp'], axis=1, inplace=True)
    return (df,responses)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    x, y = load_data('/home/ronyzerkavod/IML.HUJI/datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    df_curr =  pd.concat([x,y], axis=1)
    only_israel = df_curr[df_curr['Country']=='Israel']
    only_israel['Year'] = only_israel['Year'].astype(str)
    px.scatter(only_israel, x="DayOfYear", y="Temp", color="Year", title="Tempreture In Israel").show()
    df_Month = df_curr.groupby('Month').Temp.agg(std="std")
    px.bar(df_Month, x=df_Month.index, y="std", title="standard deviation of the daily temperatures").show()


    # Question 3 - Exploring differences between countries
    df_curr =  pd.concat([x,y], axis=1)
    df_set_month_country = df_curr.groupby(['Month','Country']).Temp.agg(std="std",mean="mean").reset_index()
    px.line(df_set_month_country, x="Month", y="mean",error_y="std", color="Country", title="the average monthly temperature and std").show()

    # Question 4 - Fitting model for different values of `k`
    israel_y = only_israel['Temp']
    israel_x = only_israel.drop(['Temp'], axis=1, inplace=False)
    train_x_i, train_y_i, test_x_i, test_y_i = split_train_test(israel_x['DayOfYear'],israel_y,0.75)
    train_x_i = np.array(train_x_i)
    train_y_i = np.array(train_y_i)
    test_x_i = np.array(test_x_i)
    test_y_i= np.array(test_y_i)
    k_values = np.linspace(1, 10, 10).astype(int)
    losses= []
    for deg in k_values:
        pol_fit = PolynomialFitting(deg)
        pol_fit.fit(train_x_i,train_y_i)
        mse_loss = pol_fit.loss(test_x_i,test_y_i)
        losses.append(round(mse_loss,2))
    print(losses)
    df_k_loss = pd.DataFrame({"k_val":k_values,"loss":losses})
    px.bar(df_k_loss, x=k_values, y=losses, title="the test error recorded for each value of k").show()  

    # Question 5 - Evaluating fitted model on different countries
    k = 5
    pol_fit_model = PolynomialFitting(5)
    pol_fit_model._fit(israel_x["DayOfYear"].to_numpy(),israel_y)
    all_countries = pd.concat([x,y],axis=1)
    df_without_israel = all_countries[all_countries['Country']!='Israel']
    all_countries_uniq = df_without_israel["Country"].unique()
    losses = []
    for country in all_countries_uniq:
        df_country = all_countries[all_countries['Country']==country]
        df_country_y = df_country['Temp']
        loss = pol_fit_model._loss(df_country["DayOfYear"],df_country_y)
        losses.append(loss)
    losses= np.array(losses)
    df_final = pd.DataFrame({"country":all_countries_uniq,"loss":losses})
    px.bar(df_final, x="country", y="loss", color="country",title="Israel's model error over each of the other countries").show()  



