"""GOF = Goodness of Fit metric

Examples:
  >>> rmse([1,2,3,4], [4,5,6,7])
  2
  >>> log_loss([[0, 1],[0, 1]], [[.6,.4],[.25,.75]])  # doctest: +ELLIPSIS
  1.20397...
  >>> llfun([[0, 1],[0, 1]], [[.6,.4],[.25,.75]])  # doctest: +ELLIPSIS
  1.20397...
  >>> metrics.log_loss([1,1], [[.6,.4],[.25,.75]])  # doctest: +ELLIPSIS
  1.20397...
"""

from scikitlearn import metrics
import pug.invest.util
rmse = pug.invest.util.rmse
import scipy as sp
from pandas import np


def log_loss(actual, predicted):
    """Log of the loss (error) summed over all entries

    The negative of the logarithm of the frequency (probability) of the predicted
    label given the true binary label for a category.

    Arguments:
      predicted (np.array of float): 2-D table of probabilities for each
        category (columns) and each record (rows)
      actual (np.array of float): True binary labels for each category
        Should only have a single 1 on each row indicating the one
        correct category (column)

    Based On:
        https://www.kaggle.com/wiki/LogarithmicLoss
        http://scikit-learn.org/stable/modules/model_evaluation.html#log-loss
    """

    predicted, actual = np.array(predicted), np.array(actual)

    small_value = 1e-15
    predicted[predicted < small_value] = small_value
    predicted[predicted > 1 - small_value] = 1. - small_value
    return (-1. / len(actual)) * np.sum(
        actual * np.log(predicted) + (1. - actual) * np.log(1. - predicted))


def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll
