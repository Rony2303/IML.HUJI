from cmath import log
from turtle import color
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu=10
    sigma=1
    X = np.random.normal(mu,sigma,1000)
    UnivariateGaussian_obj= UnivariateGaussian()
    UnivariateGaussian_obj.fit(X)
    print(UnivariateGaussian_obj.mu_,UnivariateGaussian_obj.var_)

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)
    new_estimated_mean=[]
    new_estimated_var =[]
    for i in ms:
        S= X[:i]
        new_estimated_mean.append(np.abs(UnivariateGaussian_obj.fit(S).mu_-mu))
    go.Figure([go.Scatter(x=ms, y=new_estimated_mean, mode='markers+lines', name="difference", showlegend=True)],
          layout=go.Layout(title=r"Absolute Difference Between Estimations and True Values", 
                  xaxis_title=" number of samples", 
                  yaxis_title="number of diffrence",
                  height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    dens_arr_x = UnivariateGaussian_obj.pdf(X)
    go.Figure([go.Scatter(x=X, y=dens_arr_x, mode='markers')],
          layout=go.Layout(title="Estimated Density Model Of Taken Samples", 
                  xaxis_title="Samples Values", 
                  yaxis_title="PDFs Values",
                  height=300)).show()

def test_multivariate_gaussian():

    # Question 4 - Draw samples and print fitted model
    mu = [0,0,4,0]
    sigma = [[1,0.2,0,0.5],
            [0.2,2,0,0],
            [0,0,1,0],
            [0.5,0,0,1]]
    X = np.random.multivariate_normal(mu,sigma,1000)
    MultivariateGaussian_obj = MultivariateGaussian()
    MultivariateGaussian_obj.fit(X)
    print(MultivariateGaussian_obj.mu_ )
    print(MultivariateGaussian_obj.cov_)

    # Question 5 - Likelihood evaluation
    f = np.linspace(-10, 10, 200)
    all_possible_mu = np.array(np.meshgrid(f,0,f,0)).T.reshape(-1,4) 
    fun = lambda m: MultivariateGaussian.log_likelihood(m, sigma, X)
    all_log_likelihoods = np.apply_along_axis(fun, 1, all_possible_mu).reshape(200,200)
    go.Figure(data=[go.Heatmap(x=f, y= f, z= all_log_likelihoods, type= 'heatmap')],
    layout=go.Layout(title="Heatmap Of All Log-Likelyhood Of Taken Samples With mu = [f1,0,f3,0] Values ", 
                  xaxis_title="f1 Values", 
                  yaxis_title="f3 Values",
                  height=600)).show()

    # Question 6 - Maximum likelihood
    
    print("f1 is %.3f" %all_possible_mu[np.argmax(all_log_likelihoods)][0])
    print("f3 is %.3f" % all_possible_mu[np.argmax(all_log_likelihoods)][2])

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
