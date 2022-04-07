#from asyncio.windows_events import NULL
from audioop import add
from re import X
from turtle import title
from urllib import response
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import datetime
pio.templates.default = "simple_white"

def check_conditions(X: pd.DataFrame) -> X:
    X = X.drop(X.index[X['floors'] <= 0])
    X = X.drop(X.index[X['sqft_lot'] <= 0])
    X = X.drop(X.index[X['bedrooms'] <= 0])
    X = X.drop(X.index[X['sqft_living'] < 0])
    return X


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    DF = pd.read_csv(filename)
    DF.dropna(inplace=True)
    DF = check_conditions(DF)
    # DF = pd.get_dummies(DF, columns=['zipcode'])
    responses = DF['price']
    DF.drop(['id','date','price','lat','long','zipcode'], axis=1, inplace=True)
    return (DF,responses)



def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for title in X.columns:
        pears_corr = np.cov(X[title],y)[0][1]/(np.std(X[title])*np.std(y))
        go.Figure([go.Scatter(x=X[title], y=y, mode='markers')],
          layout=go.Layout(title= f"The Conecction {title} - House Prices, Pearson Correlation {pears_corr}", 
                  xaxis_title=f"{title}", 
                  yaxis_title="house prices")).write_image(output_path+ title +".png")



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    (df,y) = load_data('/home/ronyzerkavod/IML.HUJI/datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df,y,'/home/ronyzerkavod/IML.HUJI/exercises/figures/')
    # Question 3 - Split samples into training- and testing sets.
    train_proportion = 0.75
    train_x,train_y,test_x,test_y = split_train_test(df,y,train_proportion)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    ms = np.linspace(10, 100, 91).astype(int)
    #train_x_size =len(train_x)
    all_mean = []
    all_std = []
    for i in ms:
        losses = []
        curr_perc = i/100
        for i in range(10):
            new_train_x = train_x.sample(frac = curr_perc)
            new_train_y = train_y.loc[new_train_x.index]
            new_train_y= np.array(new_train_y)
            new_train_x= np.array(new_train_x)
            linear_reg = LinearRegression()
            linear_reg._fit(new_train_x,new_train_y)
            curr_loss = linear_reg._loss(test_x.to_numpy(),test_y.to_numpy())
            losses.append(curr_loss)
        losses = np.array(losses)
        all_mean.append(losses.mean())
        all_std.append(losses.std())
    all_mean= np.array(all_mean)
    all_std = np.array(all_std)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ms, y=all_mean, mode="markers+lines", name="Mean Loss", line=dict(dash="dash"), marker=dict(color="green", opacity=.7)))
    fig.add_trace(go.Scatter(x=ms, y=all_mean-2*all_std, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False))
    fig.add_trace(go.Scatter(x=ms, y=all_mean+2*all_std, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False))
    fig.update_layout(title = "the Loss and variance of samples trained on",
                        xaxis_title="the percentages taken of the train_set",
                        yaxis_title="the MSE Losses")
    fig.show()




        
            
        
