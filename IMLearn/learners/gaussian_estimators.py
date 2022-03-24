from __future__ import annotations
import re
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None
        

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        m = X.size
        self.mu_= np.mean(X)
        if (self.biased_):
            self.var_ = (1/m)*(np.sum((X- self.mu_)**2))
        else:
            self.var_ = (1/(m-1))*(np.sum((X- self.mu_)**2))
        self.fitted_ = True
        return self

    def density(self, x:float) -> float:
        """
        calculate the density of a sample X
        returns the density
        """
        dens_of_x = (1/(np.sqrt(2*np.pi*self.var_)))*np.exp((x-self.mu_)**2/(-2*self.var_))
        return dens_of_x


    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        pdf_arr= np.ndarray(X.size)
        for i in range (X.size):
            pdf_arr[i]= self.density(X[i])
        return pdf_arr


    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        coefficient = 0 - (X.size / 2) * np.log(2 * np.pi) - (X.size / 2) * np.log(sigma)
        log_likelihood = coefficient - (1/(2 * sigma)) * np.sum((X - mu)**2)
        return log_likelihood        


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_= np.mean(X,axis=0)
        self.cov_= np.cov(X.T)
        self.fitted_ = True
        return self

    def density(self, X: np.ndarray) -> float:
        """
        calculate the pdf of vector X
        returns the pdf
        """
        d = X.size
        det = np.linalg.det(self.cov_)
        cov_inv = np.linalg.inv(self.cov_)
        ret_dens = (1/(np.sqrt((2*np.pi)**d)* det))* np.exp(-0.5*(X-self.mu_).T*cov_inv*(X-self.mu_))
        return ret_dens

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        pdf_arr= np.ndarray(X.size)
        for i in range (X.size):
            pdf_arr[i]= self.density(X[i])
        return pdf_arr
        raise NotImplementedError()

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        m = X.size
        d = X[0].size
        det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)
        log_likelihood = -((d*m)/2)*(np.log(2*np.pi)) - (m/2)*np.log(det) - (1/2)*np.sum((X- mu)@cov_inv@(X-mu).T)
        return log_likelihood





