from copy import deepcopy
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.decomposition._pca import *
from sklearn.decomposition._base import *
from sklearn.utils import check_random_state, check_scalar
from sklearn.utils.extmath import fast_logdet, svd_flip
from math import log
import numbers

from scipy.sparse import issparse
from scipy import linalg


class PCAAlpha(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        n_components=None,
        *,
        copy=True,
        center=True,
        precenter=False,
        prereduce=False,
        do_inverse=False,
        alpha=1.,
        n_oversamples=10,
        random_state=None,
    ):
        self.n_components = n_components
        self.copy = copy
        self.center = center
        self.precenter = precenter
        self.prereduce = prereduce
        self.do_inverse = do_inverse
        self.alpha = alpha
        self.n_oversamples = n_oversamples
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Ignored.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        check_scalar(
            self.n_oversamples,
            "n_oversamples",
            min_val=1,
            target_type=numbers.Integral,
        )

        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Ignored.
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        Notes
        -----
        This method returns a Fortran-ordered array. To convert it to a
        C-ordered array, use 'np.ascontiguousarray'.
        """

        
        X_ = self._fit(X)
        X_ = X_[:, :self.n_components_]

        return X_

    def _fit(self, X):

        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        if issparse(X):
            raise TypeError(
                "PCAAlpha does not support sparse input. See "
                "TruncatedSVD for a possible alternative."
            )
        
        if torch.is_tensor(X):
            self.is_torch = True
            self.phi_function = self._phi_alpha_torch
            if self.copy:
                assert len(X.size()) == 2, "X must be 2d tensor"
                X = deepcopy(X)
        else:
            self.is_torch = False
            self.phi_function = self._phi_alpha

            # Validate data
            X = self._validate_data(
                X, dtype=[np.float64, np.float32], ensure_2d=True, copy=self.copy
            )

        # Handle n_components==None
        if self.n_components is None:
            n_components = min(X.shape)
        else:
            n_components = self.n_components

        return self._fit_model(X, n_components)

    def _phi_alpha(self, x, is_inverse=False):
        '''
        :param x input matrix
        :param is_inverse (boolean) function _phi_alpha or _phi_alpha^-1 ?
        '''
        if self.alpha == 1:
            return x
        alpha = 1/self.alpha if is_inverse else self.alpha
        return np.sign(x) * np.abs(x)**alpha
    
    def _phi_alpha_torch(self, x, is_inverse=False):
        '''
        :param x input matrix
        :param is_inverse (boolean) function _phi_alpha or _phi_alpha^-1 ?
        '''
        if self.alpha == 1:
            return x
        alpha = 1/self.alpha if is_inverse else self.alpha
        return torch.sign(x) * torch.abs(x)**alpha
    
    def _numpy_decomposition(self, X, n_samples):
        
        n, d = X.shape

        # Center data
        X = self.phi_function(X)
        self.mean_ = X.mean(axis=0) if self.center else 0
        X = X - self.mean_

        U, S, Vt = linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)
        
        if n < d:
            X_ = self.phi_function(X @ Vt.T, is_inverse=True)
        else:
            X_ = self.phi_function(X @ Vt, is_inverse=True)

        return X_, Vt

    def _torch_decomposition(self, X, n_samples):
        
        n, d = X.size()

        # Center data
        X = self.phi_function(X)
        self.mean_ = X.mean(dim=0, keepdim=True) if self.center else 0
        X = X - self.mean_

        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        Vt = torch.real(Vt)
        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = self._torch_svd_flip(U, Vt, u_based_decision=True)

        if n < d:
            X_ = self.phi_function(X @ Vt.T, is_inverse=True)
        else:
            X_ = self.phi_function(X @ Vt, is_inverse=True)
        return X_, Vt

    def _fit_model(self, X, n_components):
        
        """Fit the model."""
        n_samples, n_features = X.shape

        if not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError(
                "n_components=%r must be between 0 and "
                "min(n_samples, n_features)=%r" % (n_components, min(n_samples, n_features))
            )
        elif n_components >= 1:
            if not isinstance(n_components, numbers.Integral):
                raise ValueError(
                    "n_components=%r must be of type int "
                    "when greater than or equal to 1, "
                    "was of type=%r" % (n_components, type(n_components))
                )

        # Projection & components
        if self.is_torch:
            self.premean_ = X.mean(dim=0, keepdim=True) if self.precenter else 0
            self.prestd_ = X.std(dim=0, keepdim=True) if self.prereduce else 1
            X = (X - self.premean_) / self.prestd_
            X_, components_ = self._torch_decomposition(X, n_samples)
        else:
            self.premean_ = X.mean(axis=0) if self.precenter else 0
            self.prestd_ = X.std(axis=0) if self.prereduce else 1
            X = (X - self.premean_) / self.prestd_
            X_, components_ = self._numpy_decomposition(X, n_samples)

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components

        return X_
 
    def transform(self, X):
        """Apply dimensionality reduction to X.
        X is projected on the first principal components previously extracted
        from a training set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Projection of X in the first principal components, where `n_samples`
            is the number of samples and `n_components` is the number of the components.
        """
        check_is_fitted(self)

        if not self.is_torch:
            X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)
        else:
            assert len(X.size()) == 2, "X must be 2d tensor"

        X = X - self.premean_
        X = X / self.prestd_
        X = self.phi_function(X) - self.mean_
        X_transformed = self.phi_function(X @ self.components_.T, is_inverse=True)
        return X_transformed

    def inverse_transform(self, X):
        
        X = self.phi_function(X)
        if not self.do_inverse:
            return self.phi_function(X @ self.components_ + self.mean_, is_inverse=True)*self.prestd_ + self.premean_

        pinv = torch.linalg.pinv if self.is_torch else np.linalg.pinv
        return self.phi_function(X @ pinv(self.components_.T) + self.mean_, is_inverse=True)*self.prestd_ + self.premean_

    def approximate(self, X):
        return self.inverse_transform(self.transform(X))

    def score_samples(self, X):
        """Return the log-likelihood of each sample.
        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.
        Returns
        -------
        ll : ndarray of shape (n_samples,)
            Log-likelihood of each sample under the current model.
        """
        check_is_fitted(self)
        if self.is_torch:
            raise "Not compatible with torch.tensor, use numpy instead"

        X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)
        Xr = X - self.mean_
        n_features = X.shape[1]
        precision = self.get_precision()
        log_like = -0.5 * (Xr * (np.dot(Xr, precision))).sum(axis=1)
        log_like -= 0.5 * (n_features * log(2.0 * np.pi) - fast_logdet(precision))
        return log_like

    def score(self, X, y=None):
        """Return the average log-likelihood of all samples.
        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.
        y : Ignored
            Ignored.
        Returns
        -------
        ll : float
            Average log-likelihood of the samples under the current model.
        """
        return np.mean(self.score_samples(X))

    def _more_tags(self):
        return {"preserves_dtype": [np.float64, np.float32]}

    def _torch_svd_flip(self, u, v, u_based_decision=True):
        if u_based_decision:
            # columns of u, rows of v
            max_abs_cols = torch.argmax(u.abs(), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
            u *= signs
            v *= signs.unsqueeze(-1)
        else:
            # rows of v, columns of u
            max_abs_rows = torch.argmax(v.abs(), dim=-1)
            signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
            u *= signs
            v *= signs.unsqueeze(-1)
        return u, v



class PLSAlpha(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        n_components=None,
        *,
        copy=True,
        center=True,
        do_inverse=False,
        alpha=1.,
        n_oversamples=10,
        random_state=None,
    ):
        self.n_components = n_components
        self.copy = copy
        self.center = center
        self.do_inverse = do_inverse
        self.alpha = alpha
        self.n_oversamples = n_oversamples
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Ignored.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        check_scalar(
            self.n_oversamples,
            "n_oversamples",
            min_val=1,
            target_type=numbers.Integral,
        )

        self._fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Ignored.
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        Notes
        -----
        This method returns a Fortran-ordered array. To convert it to a
        C-ordered array, use 'np.ascontiguousarray'.
        """

        
        X_ = self._fit(X, y)
        X_ = X_[:, :self.n_components_]

        return X_

    def _fit(self, X, y):

        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        if issparse(X):
            raise TypeError(
                "PCAAlpha does not support sparse input. See "
                "TruncatedSVD for a possible alternative."
            )
        
        if torch.is_tensor(X):
            self.is_torch = True
            self.phi_function = self._phi_alpha_torch
            if self.copy:
                assert len(X.size()) == 2, "X must be 2d tensor"
                X = deepcopy(X)
                if y is not None:
                    y = deepcopy(y)
                    if len(y.size()) == 1: y = y.unsqueeze(-1)
        else:
            self.is_torch = False
            self.phi_function = self._phi_alpha

            # Validate data
            X = self._validate_data(
                X, dtype=[np.float64, np.float32], ensure_2d=True, copy=self.copy
            )
            if y is not None:
                y = check_array(
                y, dtype=np.float64, copy=self.copy, ensure_2d=False
                )
                if y.ndim == 1:
                    y = y.reshape(-1, 1)
            

        # Handle n_components==None
        if self.n_components is None:
            n_components = min(X.shape)
        else:
            n_components = self.n_components

        return self._fit_model(X, y, n_components)

    def _phi_alpha(self, x, is_inverse=False):
        '''
        :param x input matrix
        :param is_inverse (boolean) function _phi_alpha or _phi_alpha^-1 ?
        '''
        alpha = 1/self.alpha if is_inverse else self.alpha
        return np.sign(x) * np.abs(x)**alpha
    
    def _phi_alpha_torch(self, x, is_inverse=False):
        '''
        :param x input matrix
        :param is_inverse (boolean) function _phi_alpha or _phi_alpha^-1 ?
        '''
        alpha = 1/self.alpha if is_inverse else self.alpha
        return torch.sign(x) * torch.abs(x)**alpha
    
    def _numpy_decomposition(self, X, y, n_samples):
        
        # Center data
        X = self.phi_function(X)
        self.mean_ = X.mean(axis=0) if self.center else 0
        X = X - self.mean_

        if y is not None:
            y = self.phi_function(y)
            y = y - y.mean(axis=0)
            cov = X.T @ y @ y.T @ X / n_samples
        else:
            cov = X.T @ X / n_samples

        U, S, Vt = linalg.svd(cov, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)
        X_ = self.phi_function(X @ Vt, is_inverse=True)

        return X_, Vt

    def _torch_decomposition(self, X, y, n_samples):
        
        # Center data
        X = self.phi_function(X)
        self.mean_ = X.mean(dim=0, keepdim=True) if self.center else 0
        X = X - self.mean_

        if y is not None:
            y = self.phi_function(y)
            y = y - y.mean(dim=0, keepdim=True)
            cov = X.T @ y @ y.T @ X / n_samples
        else:
            cov = X.T @ X / n_samples

        valp, Vt = torch.linalg.eig(cov)
        Vt = torch.real(Vt).T
        X_ = self.phi_function(X @ Vt, is_inverse=True)

        return X_, Vt

    def _fit_model(self, X, y, n_components):
        
        """Fit the model."""
        n_samples, n_features = X.shape

        if not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError(
                "n_components=%r must be between 0 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='full'" % (n_components, min(n_samples, n_features))
            )
        elif n_components >= 1:
            if not isinstance(n_components, numbers.Integral):
                raise ValueError(
                    "n_components=%r must be of type int "
                    "when greater than or equal to 1, "
                    "was of type=%r" % (n_components, type(n_components))
                )

        # Projection & components
        if self.is_torch:
            X_, components_ = self._torch_decomposition(X, y, n_samples)
        else:
            X_, components_ = self._numpy_decomposition(X, y, n_samples)

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components

        return X_
 
    def transform(self, X):
        """Apply dimensionality reduction to X.
        X is projected on the first principal components previously extracted
        from a training set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Projection of X in the first principal components, where `n_samples`
            is the number of samples and `n_components` is the number of the components.
        """
        check_is_fitted(self)

        if not self.is_torch:
            X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)
        else:
            assert len(X.size()) == 2, "X must be 2d tensor"

        X = self.phi_function(X) - self.mean_
        X_transformed = self.phi_function(X @ self.components_.T, is_inverse=True)
        return X_transformed

    def inverse_transform(self, X):
        
        X = self.phi_function(X)
        if not self.do_inverse:
            return self.phi_function(X @ self.components_ + self.mean_, is_inverse=True)

        pinv = torch.linalg.pinv if self.is_torch else np.linalg.pinv
        return self.phi_function(X @ pinv(self.components_.T) + self.mean_, is_inverse=True)

    def approximate(self, X):
        return self.inverse_transform(self.transform(X))

    def score_samples(self, X):
        """Return the log-likelihood of each sample.
        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.
        Returns
        -------
        ll : ndarray of shape (n_samples,)
            Log-likelihood of each sample under the current model.
        """
        check_is_fitted(self)
        if self.is_torch:
            raise "Not compatible with torch.tensor, use numpy instead"

        X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)
        Xr = X - self.mean_
        n_features = X.shape[1]
        precision = self.get_precision()
        log_like = -0.5 * (Xr * (np.dot(Xr, precision))).sum(axis=1)
        log_like -= 0.5 * (n_features * log(2.0 * np.pi) - fast_logdet(precision))
        return log_like

    def score(self, X, y=None):
        """Return the average log-likelihood of all samples.
        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.
        y : Ignored
            Ignored.
        Returns
        -------
        ll : float
            Average log-likelihood of the samples under the current model.
        """
        return np.mean(self.score_samples(X))

    def _more_tags(self):
        return {"preserves_dtype": [np.float64, np.float32]}

