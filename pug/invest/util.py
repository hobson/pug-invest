# util.py
from __future__ import print_function

from collections import Mapping, OrderedDict
import datetime
import itertools
import random
import warnings

import pandas as pd
np = pd.np

from scipy import integrate

from matplotlib import pyplot as plt
import seaborn
from scipy.optimize import minimize
from scipy.signal import correlate
from titlecase import titlecase


from pug.nlp.util import listify, fuzzy_get, make_timestamp


def dropna(x):
    """Delete all NaNs and and infinities in a sequence of real values

    Returns:
        list: Array of all values in x that are between -inf and +inf, exclusive
    """
    return [x_i for x_i in listify(x) if float('-inf') < x_i < float('inf')]


def rms(x):
    """"Root Mean Square"

    Arguments:
        x (seq of float): A sequence of numerical values

    Returns:
        The square root of the average of the squares of the values

        math.sqrt(sum(x_i**2 for x_i in x) / len(x))

        or

        return (np.array(x) ** 2).mean() ** 0.5

    >>> rms([0, 2, 4, 4])
    3.0
    """
    try:
        return (np.array(x) ** 2).mean() ** 0.5
    except:
        x = np.array(dropna(x))
        invN = 1.0 / len(x)
        return (sum(invN * (x_i ** 2) for x_i in x)) ** .5


def rmse(target, prediction, relative=False, percent=False):
    """Root Mean Square Error

    This seems like a simple formula that you'd never need to create a function for.
    But my mistakes on coding challenges have convinced me that I do need it,
    as a reminder of important tweaks, if nothing else.

    >>> rmse([0, 1, 4, 3], [2, 1, 0, -1])
    3.0
    >>> rmse([0, 1, 4, 3], [2, 1, 0, -1], relative=True)  # doctest: +ELLIPSIS
    1.2247...
    >>> rmse([0, 1, 4, 3], [2, 1, 0, -1], percent=True)  # doctest: +ELLIPSIS
    122.47...
    """
    relative = relative or percent
    prediction = pd.np.array(prediction)
    target = np.array(target)
    err = prediction - target
    if relative:
        denom = target
        # Avoid ZeroDivisionError: divide by prediction rather than target where target==0
        denom[denom == 0] = prediction[denom == 0]
        # If the prediction and target are both 0, then the error is 0 and should be included in the RMSE
        # Otherwise, the np.isinf() below would remove all these zero-error predictions from the array.
        denom[(denom == 0) & (target == 0)] = 1
        err = (err / denom)
        err = err[(~ np.isnan(err)) & (~ np.isinf(err))]
    return 100 * rms(err) if percent else rms(err)


def blended_rolling_apply(series, window=2, fun=pd.np.mean):
    new_series = pd.Series(np.fromiter((fun(series[:i + 1]) for i in range(window - 1)),
                           type(series.values[0])), index=series.index[:window - 1]).append(
                               pd.rolling_apply(series.copy(), window, fun)[window - 1:])
    assert len(series) == len(new_series), (
        "blended_rolling_apply should always return a series of the same length!\n"
        " len(series) = {0} != {1} = len(new_series".format(len(series), len(new_series)))
    assert not any(np.isnan(val) or val is None for val in new_series)
    return new_series


def rolling_latch(series, period=31, decay=1.0):
    # FIXME: implement recursive exponential decay filter rather than the nonrecursive, deratring done here
    return blended_rolling_apply(series, period, lambda val: decay * pd.np.max(val))


def clean_dataframe(df):
    """Fill NaNs with the previous value, the next value or if all are NaN then 1.0"""
    df = df.fillna(method='ffill')
    df = df.fillna(0.0)
    return df


def clean_dataframes(dfs):
    """Fill NaNs with the previous value, the next value or if all are NaN then 1.0

    TODO:
      Linear interpolation and extrapolation

    Arguments:
      dfs (list of dataframes): list of dataframes that contain NaNs to be removed

    Returns:
      list of dataframes: list of dataframes with NaNs replaced by interpolated values
    """
    if isinstance(dfs, (list)):
        for df in dfs:
            df = clean_dataframe(df)
        return dfs
    else:
        return [clean_dataframe(dfs)]


def get_symbols_from_list(list_name):
    """Retrieve a named (symbol list name) list of strings (symbols)

    If you've installed the QSTK Quantitative analysis toolkit
        `get_symbols_from_list('sp5002012')` will produce a list of the symbols that
        were members of the S&P 500 in 2012.
    Otherwise an import error exception will be raised.
        If the symbol list cannot be found you'll get an empty list returned

    Example:
      >> len(get_symbols_from_list('sp5002012')) in (0, 501)
      True
    """
    try:
        # quant software toolkit has a method for retrieving lists of symbols like S&P500 for 2012 with 'sp5002012'
        import QSTK.qstkutil.DataAccess as da
        dataobj = da.DataAccess('Yahoo')
    except ImportError:
        raise
    except:
        return []
    try:
        return dataobj.get_symbols_from_list(list_name)
    except:
        raise


def make_symbols(symbols, *args):
    """Return a list of uppercase strings like "GOOG", "$SPX, "XOM"...

    Arguments:
      symbols (str or list of str): list of market ticker symbols to normalize
        If `symbols` is a str a get_symbols_from_list() call is used to retrieve the list of symbols

    Returns:
      list of str: list of cananical ticker symbol strings (typically after .upper().strip())

    See Also:
      pug.dj.db.normalize_names

    Examples:
      >>> make_symbols("Goog")
      ['GOOG']
      >>> make_symbols("  $SPX   ", " aaPL ")
      ['$SPX', 'AAPL']
      >>> make_symbols(["$SPX", ["GOOG", "AAPL"]])
      ['GOOG', 'AAPL', '$SPX']
      >>> make_symbols(" $Spy, Goog, aAPL ")
      ['$SPY', 'GOOG', 'AAPL']
    """
    if (hasattr(symbols, '__iter__') and not any(symbols)) \
            or (isinstance(symbols, (list, tuple, Mapping)) and not symbols):
        return []
    if isinstance(symbols, basestring):
        # # FIXME: find a direct API for listing all possible symbols
        # try:
        #     return list(set(dataobj.get_symbols_from_list(symbols)))
        # except:
        return [s.upper().strip() for s in (symbols.split(',') + list(str(a) for a in args))]
    else:
        ans = []
        for sym in (list(symbols) + list(args)):
            tmp = make_symbols(sym)
            ans = ans + tmp
        return list(set(ans))


def make_time_series(x, t=pd.Timestamp(datetime.datetime(1970, 1, 1)), freq=None):
    """Convert a 2-D array of time/value pairs (or pair of time/value vectors) into a pd.Series time-series

    >>> make_time_series(range(3), freq='15min')  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    1970-01-01 00:00:00   NaN
    1970-01-01 00:15:00   NaN
    1970-01-01 00:30:00   NaN
    dtype: float64
    """
    if isinstance(x, pd.DataFrame):
        x = pd.Series(x[x.columns[0]])
    elif not isinstance(x, pd.Series) and (not isinstance(t, (pd.Series, pd.Index, list, tuple)) or not len(t)):
        #warnings.warn("Coercing a non-Series")
        if len(x) == 2:
            t, x = listify(x[0]), listify(x[1])
        elif len(x) >= 2:
            try:
                t, x = zip(*x)
            except (ValueError, IndexError, TypeError):
                pass
        x = pd.Series(x)
    else:
        if isinstance(t, (datetime.datetime, pd.Timestamp)):
            t = pd.Timestamp(t)
        else:
            x = pd.Series(listify(x), index=listify(t))
    if not isinstance(x, pd.Series):
        raise TypeError("`pug.invest.util.make_time_series(x, t)` expects x to be a type that"
                        " can be coerced to a Series object, but it's type is: {0}"
                        .format(type(x)))
    # By this point x must be a Series, only question is whether its index needs to be converted to a DatetimeIndex
    if x.index[0] != 0 and isinstance(x.index[0], (datetime.date, datetime.datetime, pd.Timestamp,
                                      basestring, float, np.int64, int)):
        t = x.index
    elif isinstance(t, (datetime.date, datetime.datetime, pd.Timestamp, basestring, float, np.int64, int)):
        if not freq:
            freq = '15min'
            warnings.warn('Assumed time series freq to be {0} though no freq argument was provided!'
                          .format(freq), RuntimeWarning)
        t = pd.date_range(t, periods=len(x), freq=freq)
    x = pd.Series(x, index=t)
    if isinstance(x, pd.Series):
        x.index = pd.DatetimeIndex(x.index.values)
    return x


def pandas_mesh(df):
    """Create numpy 2-D "meshgrid" from 3+ columns in a Pandas DataFrame

    Arguments:
      df (DataFrame): Must have 3 or 4 columns of numerical data

    Returns:
      OrderedDict: column labels from the data frame are the keys, values are 2-D matrices
        All matrices have shape NxM, where N = len(set(df.iloc[:,0])) and M = len(set(df.iloc[:,1]))

    >>> pandas_mesh(pd.DataFrame(np.arange(18).reshape(3,6),
    ...                          columns=list('ABCDEF'))).values()  # doctest: +NORMALIZE_WHITESPACE
    [array([[ 0,  6, 12],
            [ 0,  6, 12],
            [ 0,  6, 12]]),
     array([[ 1,  1,  1],
            [ 7,  7,  7],
            [13, 13, 13]]),
     array([[  2.,  nan,  nan],
            [ nan,   8.,  nan],
            [ nan,  nan,  14.]]),
     array([[  3.,  nan,  nan],
            [ nan,   9.,  nan],
            [ nan,  nan,  15.]]),
     array([[  4.,  nan,  nan],
            [ nan,  10.,  nan],
            [ nan,  nan,  16.]]),
     array([[  5.,  nan,  nan],
            [ nan,  11.,  nan],
            [ nan,  nan,  17.]])]
    """
    xyz = [df[c].values for c in df.columns]
    index = pd.MultiIndex.from_tuples(zip(xyz[0], xyz[1]), names=['x', 'y'])
    # print(index)
    series = [pd.Series(values, index=index) for values in xyz[2:]]
    # print(series)
    X, Y = np.meshgrid(sorted(list(set(xyz[0]))), sorted(list(set(xyz[1]))))
    N, M = X.shape
    Zs = []
    # print(Zs)
    for k, s in enumerate(series):
        Z = np.empty(X.shape)
        Z[:] = np.nan
        for i, j in itertools.product(range(N), range(M)):
            Z[i, j] = s.get((X[i, j], Y[i, j]), np.NAN)
        Zs += [Z]
    return OrderedDict((df.columns[i], m) for i, m in enumerate([X, Y] + Zs))


def integrated_change(ts, integrator=integrate.trapz, clip_floor=None, clip_ceil=float('inf')):
    """Total value * time above the starting value within a TimeSeries"""
    integrator = get_integrator(integrator)
    if clip_floor is None:
        clip_floor = ts[0]
    if clip_ceil < clip_floor:
        polarity = -1
        offset, clip_floor, clip_ceil, = clip_ceil, clip_ceil, clip_floor
    else:
        polarity, offset = 1, clip_floor
    clipped_values = np.clip(ts.values - offset, clip_floor, clip_ceil)
    print(polarity, offset, clip_floor, clip_ceil)
    print(clipped_values)
    integrator_types = set(['trapz', 'cumtrapz', 'simps', 'romb'])
    if integrator in integrator_types:
        integrator = getattr(integrate, integrator)
    integrator = integrator or integrate.trapz
    # datetime units converted to seconds (since 1/1/1970)
    return integrator(clipped_values, ts.index.astype(np.int64) / 10 ** 9)


def insert_crossings(ts, thresh):
    """Insert/append threshold crossing points (time and value) into a timeseries (pd.Series)

    Arguments:
      ts (pandas.Series): Time series of values to be interpolated at `thresh` crossings
      thresh (float or np.float64):
    """
    # import time
    # tic0 = time.clock(); tic = tic0

    # int64 for fast processing, pandas.DatetimeIndex is 5-10x slower, 0.3 ms
    index = ts.index
    index_type = type(index)
    ts.index = ts.index.astype(np.int64)
    # toc = time.clock();
    # print((toc-tic)*1000); tic = time.clock()

    # value immediately before an upward thresh crossing, 6 ms
    preup = ts[(ts < thresh) & (ts.shift(-1) > thresh)]
    # toc = time.clock();
    # print((toc-tic)*1000); tic = time.clock()

    # values immediately after an upward thresh crossing, 4 ms\
    postup = ts[(ts.shift(1) < thresh) & (ts > thresh)]
    # toc = time.clock();
    # print((toc-tic)*1000); tic = time.clock()

    # value immediately after a downward thresh crossing, 1.8 ms
    postdown = ts[(ts < thresh) & (ts.shift(1) > thresh)]
    # toc = time.clock();
    # print((toc-tic)*1000); tic = time.clock()

    # value immediately before an upward thresh crossing, 1.9 ms
    predown = ts[(ts.shift(-1) < thresh) & (ts > thresh)]
    # toc = time.clock();
    # print((toc-tic)*1000); tic = time.clock()

    # upward slope (always positive) between preup and postup in units of
    # "value" per nanosecond (timestamps convert to floats as nanoseconds), 0.04 ms
    slopeup = (postup.values - preup.values) / (postup.index.values - preup.index.values).astype(np.float64)
    # toc = time.clock();
    # print((toc-tic)*1000); tic = time.clock()

    # upward crossing point index/time, 0.04 ms
    tup = preup.index.values + ((thresh - preup.values) / slopeup).astype(np.int64)
    # toc = time.clock();
    # print((toc-tic)*1000); tic = time.clock()

    # downward slope (always negative) between predown and postdown in units of
    # "value" per nanosecond (timestamps convert to floats as nanoseconds), 0.03 ms
    slopedown = (postdown.values - predown.values) / \
                (postdown.index.values - predown.index.values).astype(np.float64)
    # toc = time.clock();
    # print((toc-tic)*1000); tic = time.clock()

    # upward crossing point index/time, 0.02 ms
    tdown = predown.index.values + ((thresh - predown.values) / slopedown).astype(np.int64)
    # toc = time.clock();
    # print((toc-tic)*1000); tic = time.clock()

    # insert crossing points into time-series (if it had a regular sample period before, it won't now!), 2.0 ms
    ts.index = index  # pd.DatetimeIndex(ts.index)
    # toc = time.clock();
    # print((toc-tic)*1000); tic = time.clock()

    # insert crossing points into time-series (if it had a regular sample period before, it won't now!), 2.0 ms
    ts = ts.append(pd.Series(thresh * np.ones(len(tup)), index=index_type(tup.astype(np.int64))))
    # toc = time.clock();
    # print((toc-tic)*1000); tic = time.clock()

    # insert crossing points into time-series (if it had a regular sample period before, it won't now!), 1.9 ms
    ts = ts.append(pd.Series(thresh * np.ones(len(tdown)), index=index_type(tdown.astype(np.int64))))
    # toc = time.clock();
    # print((toc-tic)*1000); tic = time.clock()

    # if you don't `sort_index()`, numerical integrators in `scipy.integrate` will give the wrong answer, 0.1 ms
    ts = ts.sort_index()
    # toc = time.clock();
    # if you don't `sort_index()`, numerical integrators in `scipy.integrate` will give the wrong answer
    # print((toc-tic)*1000); tic = time.clock()
    # print((toc-tic0)*1000);
    return ts


def get_integrator(integrator):
    """Return the scipy.integrator indicated by an index, name, or integrator_function

    >> get_integrator(0)
    """
    integrator_types = set(['trapz', 'cumtrapz', 'simps', 'romb'])
    integrator_funcs = [integrate.trapz, integrate.cumtrapz, integrate.simps, integrate.romb]

    if isinstance(integrator, int) and 0 <= integrator < len(integrator_types):
        integrator = integrator_types[integrator]
    if isinstance(integrator, basestring) and integrator in integrator_types:
        return getattr(integrate, integrator)
    elif integrator in integrator_funcs:
        return integrator
    else:
        print('Unsupported integration rule: {0}'.format(integrator))
        print('Expecting one of these sample-based integration rules: %s' % (str(list(integrator_types))))
        raise AttributeError
    return integrator


def clipped_area(ts, thresh=0, integrator=integrate.trapz):
    """Total value * time above the starting value within a TimeSeries

    Arguments:
      ts (pandas.Series): Time series to be integrated.
      thresh (float): Value to clip the tops off at (crossings will be interpolated)

    References:
      http://nbviewer.ipython.org/gist/kermit666/5720498

    >>> t = ['2014-12-09T00:00', '2014-12-09T00:15', '2014-12-09T00:30', '2014-12-09T00:45',
    ...      '2014-12-09T01:00', '2014-12-09T01:15', '2014-12-09T01:30', '2014-12-09T01:45']
    >>> import pandas as pd
    >>> ts = pd.Series([217, 234, 235, 231, 219, 219, 231, 232], index=pd.to_datetime(t))
    >>> clipped_area(ts, thresh=230)  # doctest: +ELLIPSIS
    8598.52941...
    >>> clipped_area(ts, thresh=234)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    562.5
    >>> clipped_area(pd.Series(ts.values, index=ts.index.values.astype(pd.np.int64)),
    ...              thresh=234)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    562.5
    """
    integrator = get_integrator(integrator or 0)
    ts = insert_crossings(ts, thresh) - thresh
    ts = ts[ts >= 0]
    # timestamp is in nanoseconds (since 1/1/1970) but this converts it to seconds (SI units)
    return integrator(ts, ts.index.astype(np.int64)) / 1.0e9


def clipping_params(ts, capacity=100, rate_limit=float('inf'), method=None, max_attempts=100):
    """Start, end, and threshold that clips the value of a time series the most, given a limitted "capacity" and "rate"

    Assumes that signal can be linearly interpolated between points (trapezoidal integration)

    Arguments:
      ts (TimeSeries): Time series to attempt to clip to as low a max value as possible
      capacity (float): Total "funds" or "energy" available for clipping (integrated area under time series)
      method (str): scipy optimization algorithm name, one of:
        'L-BFGS-B': Byrd, 1995, "A Limited Memory Algorithm for Bound Constrained Optimization"
        'TNC':      Truncated Newton in C, or Newton Conjugate-Gradient, each variable may be constrained with upper and lower bounds
        'COBYLA':   Constrained Optimization by Linear Approximation. Fortran implementation.
        'SLSQP':    Kraft, 1988, Sequential Least Squares Programming or Quadratic Programming, infinite bounds converted to large floats

    TODO:
      Bisection search for the optimal threshold.

    Returns:
      2-tuple: Timestamp of the start and end of the period of the maximum clipped integrated increase

    >>> t = ['2014-12-09T00:00', '2014-12-09T00:15', '2014-12-09T00:30', '2014-12-09T00:45', '2014-12-09T01:00', '2014-12-09T01:15', '2014-12-09T01:30', '2014-12-09T01:45']
    >>> import pandas as pd
    >>> ts = pd.Series([217, 234, 235, 231, 219, 219, 231, 232], index=pd.to_datetime(t))  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    >>> clipping_params(ts, capacity=60000)['threshold']  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    218.13...
    >>> clipping_params(ts, capacity=30000)['threshold']  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    224.15358...
    """
    VALID_METHODS = ['L-BFGS-B', 'TNC', 'SLSQP', 'COBYLA']
    # print('in clipping params for ts.index={0} and method={1}'.format(ts.index[0], method))
    ts.index = ts.index.astype(np.int64)
    costs = []

    def cost_fun(x, *args):
        thresh = x[0]
        ts, capacity, bounds = args
        integral = clipped_area(ts, thresh=thresh)
        terms = np.array([(10. * (integral - capacity) / capacity) ** 2,
                        2. / 0.1**((bounds[0] - thresh) * capacity / bounds[0]),
                        2. / 0.1**((thresh - bounds[1]) * capacity / bounds[1]),
                        1.2 ** (integral / capacity)])
        return sum(terms)

    bounds = (ts.min(), ts.max())
    done, attempts = 0, 0
    thresh0 = bounds[0] + 0.5 * (bounds[1] - bounds[0])
    if not method or not method in VALID_METHODS:
        while attempts < max_attempts and not done:
            for optimizer_method in VALID_METHODS:
                optimum = minimize(fun=cost_fun, x0=[thresh0], bounds=[bounds], args=(ts, capacity, bounds), method=optimizer_method)
                if optimum.success:
                    done = True
                    break
            if done:
                break
            attempts += 1
            thresh0 = bounds[0] + random.random() * (bounds[1] - bounds[0])
    else:
        optimum = minimize(fun=cost_fun, x0=[thresh0], bounds=[bounds], args=(ts, capacity, bounds), method=method)
    thresh = optimum.x[0]
    integral = clipped_area(ts, thresh=thresh)
    params = dict(optimum)
    params.update({'costs': costs, 'threshold': thresh, 'initial_guess': thresh0, 'attempts': attempts,
                   'integral': integral, 'method': method})
    return params
    # if integral - capacity > capacity:
    #     return {'t0': None, 't1': None, 'threshold': 0.96*thresh + 0.06*bounds[0][1], 'integral': integral}


def discrete_clipping_params(ts, capacity=100, rate_limit=float('inf')):
    """Start, end, and threshold that clips the value of a time series the most, given a limitted "capacity" and "rate"

    Assumes that the integrated maximum includes the peak (instantaneous maximum).
    Assumes that the threshold can only set to one of the values of the Series.

    Arguments:
      ts (TimeSeries): Time series to attempt to clip to as low a max value as possible
      capacity (float): Total "funds" or "energy" available for clipping (integrated area under time series)

    TODO:
      Bisection search for the optimal threshold.

    Returns:
      2-tuple: Timestamp of the start and end of the period of the maximum clipped integrated increase

    >> t = ['2014-12-09T00:00', '2014-12-09T00:15', '2014-12-09T00:30', '2014-12-09T00:45',
    .. '2014-12-09T01:00', '2014-12-09T01:15', '2014-12-09T01:30', '2014-12-09T01:45']
    >> ts = pd.Series([217, 234, 235, 231, 219, 219, 231, 232], index=pd.to_datetime(t))
    >> (discrete_clipping_params(ts, capacity=60000) ==
    .. {'integral': 54555.882352942499, 't0': pd.Timestamp('2014-12-09 00:15:00'),
    ..  't1': pd.Timestamp('2014-12-09 01:45:00'),
    .. 'threshold': 219})
    True
    >> (discrete_clipping_params(ts, capacity=30000) ==
    .. {'integral': 5638.2352941179997, 't0': pd.Timestamp('2014-12-09 00:15:00'),
    .. 't1': pd.Timestamp('2014-12-09 01:45:00'),
    .. 'threshold': 231})
    True
    """
    raise NotImplementedError("Doesn't work. Returns incorrect, overly conservative threshold values.")
    #index_type = ts.index.dtype
    #ts2 = ts.copy()
    ts.index = ts.index.astype(np.int64)
    ts_sorted = ts.order(ascending=False)
    # default is to clip right at the peak (no clipping at all)
    i, t0, t1, integral, thresh = 1, ts_sorted.index[0], ts_sorted.index[0], 0, ts_sorted.iloc[0]
    params = {'t0': t0, 't1': t1, 'integral': 0, 'threshold': thresh}
    while i < len(ts_sorted) and integral <= capacity and (ts_sorted.iloc[0] - ts_sorted.iloc[i]) < rate_limit:
        params = {'t0': pd.Timestamp(t0), 't1': pd.Timestamp(t1), 'threshold': thresh, 'integral': integral}
        i += 1
        times = ts_sorted.index[:i]
        # print(times)
        t0 = times.min()
        t1 = times.max()
        # print(ts_sorted.index[:3])
        thresh = min(ts_sorted.iloc[:i])
        integral = clipped_area(ts, thresh=thresh)
    if integral <= capacity:
        return {'t0': pd.Timestamp(t0), 't1': pd.Timestamp(t1), 'threshold': thresh, 'integral': integral}
    return params


def square_off(series, time_delta=None, transition_seconds=1):
    """Insert samples in regularly sampled data to produce stairsteps from ramps when plotted.

    New samples are 1 second (1e9 ns) before each existing samples, to facilitate plotting and sorting

    >>> square_off(pd.Series(range(3), index=pd.date_range('2014-01-01', periods=3, freq='15m')),
    ...            time_delta=5.5)  # doctest: +NORMALIZE_WHITESPACE
    2014-01-31 00:00:00           0
    2014-01-31 00:00:05.500000    0
    2015-04-30 00:00:00           1
    2015-04-30 00:00:05.500000    1
    2016-07-31 00:00:00           2
    2016-07-31 00:00:05.500000    2
    dtype: int64
    >>> square_off(pd.Series(range(2), index=pd.date_range('2014-01-01', periods=2, freq='15min')),
    ...            transition_seconds=2.5)  # doctest: +NORMALIZE_WHITESPACE
    2014-01-01 00:00:00           0
    2014-01-01 00:14:57.500000    0
    2014-01-01 00:15:00           1
    2014-01-01 00:29:57.500000    1
    dtype: int64
    """
    if time_delta:
        # int, float means delta is in seconds (not years!)
        if isinstance(time_delta, (int, float)):
            time_delta = datetime.timedelta(0, time_delta)
        new_times = series.index + time_delta
    else:
        diff = np.diff(series.index)
        time_delta = np.append(diff, [diff[-1]])
        new_times = series.index + time_delta
        new_times = pd.DatetimeIndex(new_times) - datetime.timedelta(0, transition_seconds)
    return pd.concat([series, pd.Series(series.values, index=new_times)]).sort_index()


def clipping_threshold(ts, capacity=100, rate_limit=10):
    """Start and end index (datetime) that clips the price/value of a time series the most

    Assumes that the integrated maximum includes the peak (instantaneous maximum).

    Arguments:
      ts (TimeSeries): Time series of prices or power readings to be "clipped" as much as possible.
      capacity (float): Total "funds" or "energy" available for clipping (in $ or Joules)
        The maximum allowed integrated area under time series and above the clipping threshold.
      rate_limit: Maximum rate at which funds or energy can be expended (in $/s or Watts)
        The clipping threshold is limitted to no less than the peak power (price rate) minus this rate_limit

    Returns:
      dict: Timestamp of the start and end of the period of the maximum clipped integrated increase

    >>> t = ['2014-12-09T00:00', '2014-12-09T00:15', '2014-12-09T00:30', '2014-12-09T00:45',
    ...      '2014-12-09T01:00', '2014-12-09T01:15', '2014-12-09T01:30', '2014-12-09T01:45']
    >>> import pandas as pd
    >>> ts = pd.Series([217, 234, 235, 231, 219, 219, 231, 232], index=pd.to_datetime(t))
    >>> clipping_threshold(ts, capacity=60000)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    218.13...
    >>> clipping_threshold(ts, capacity=30000)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    224.15...
    """
    params = clipping_params(ts, capacity=capacity, rate_limit=rate_limit)
    if params:
        return params['threshold']
    return None


def join_time_series(serieses, ignore_year=False, T_s=None, aggregator='mean'):
    """Combine a dict of pd.Series objects into a single pd.DataFrame with optional downsampling

    FIXME:
      For ignore_year and multi-year data, the index (in seconds) is computed assuming
      366 days per year (leap year). So 3 out of 4 years will have a 1-day (86400 s) gap

    Arguments:
      series (dict of Series): dictionary of named timestamp-indexed Series objects
      ignore_year (bool): ignore the calendar year, but not the season (day of year)
         If True, the DataFrame index will be seconds since the beginning of the
         year in each Series index, i.e. midnight Jan 1, 2014 will have index=0
         as will Jan 1, 2010 if two Series start on those two dates.
      T_s (float): sample period in seconds (for downsampling)
      aggregator (str or func): e.g. 'mean', 'sum', np.std
    """
    if ignore_year:
        df = pd.DataFrame()
        for name, ts in serieses.iteritems():
            # FIXME: deal with leap years
            sod = np.array(map(lambda x: (x.hour * 3600 + x.minute * 60 + x.second),
                           ts.index.time))
            # Coerce soy to an integer so that merge/join operations identify same values
            # (floats don't equal!?)
            soy = (ts.index.dayofyear + 366 * (ts.index.year - ts.index.year[0])) * 3600 * 24 + sod
            ts2 = pd.Series(ts.values, index=soy)
            ts2 = ts2.dropna()
            ts2 = ts2.sort_index()
            df2 = pd.DataFrame({name: ts2.values}, index=soy)
            df = df.join(df2, how='outer')
        if T_s and aggregator:
            df = df.groupby(lambda x: int(x /
                            float(T_s))).aggregate(dict((name, aggregator) for name in df.columns))
    else:
        df = pd.DataFrame(serieses)
        if T_s and aggregator:
            x0 = df.index[0]
            df = df.groupby(lambda x: int((x - x0).total_seconds() /
                            float(T_s))).aggregate(dict((name, aggregator) for name in df.columns))
            # FIXME: convert seconds since begninning of first year back into Timestamp instances
    return df


def simulate(t=1000, poly=(0.,), sinusoids=None, sigma=0, rw=0, irw=0, rrw=0):
    """Simulate a random signal with seasonal (sinusoids), linear and quadratic trend, RW, IRW, and RRW

    Arguments:
      t (int or list of float): number of samples or time vector, default = 1000
      poly (list of float): polynomial coefficients (in decreasing "order") passed to `numpy.polyval`
         i.e. poly[0]*x**(N-1) + ... + poly[N-1]
      sinusoids (list of list): [[period], [amplitude, period], or [ampl., period, phase]]

    >>> len(simulate(poly=(0,),rrw=1))
    1000
    >>> simulate(t=range(3), poly=(1,2))  # doctest: +NORMALIZE_WHITESPACE
    0    2
    1    3
    2    4
    dtype: float64
    >>> all(simulate(t=50, sinusoids=((1,2,3),)) == simulate(t=range(50), sinusoids=((1,2,3),)))
    True
    >>> any(simulate(t=100))
    False
    >>> abs(simulate(sinusoids=42.42).values[1] + simulate(sinusoids=42.42).values[-1]) < 1e-10
    True
    >>> simulate(t=17,sinusoids=[42, 16]).min()
    -42.0
    >>> all((simulate(t=range(10), sinusoids=(1, 9, 4.5))+simulate(t=10, sinusoids=(1,9))).abs() < 1e-10)
    True
    """
    if t and isinstance(t, int):
        t = np.arange(t, dtype=np.float64)
    else:
        t = np.array(t, dtype=np.float64)
    N = len(t)
    poly = poly or (0.,)
    poly = listify(poly)
    y = np.polyval(poly, t)
    sinusoids = listify(sinusoids or [])
    if any(isinstance(ATP, (int, float)) for ATP in sinusoids):
        sinusoids = [sinusoids]
    for ATP in sinusoids:
        # default period is 1 more than the length of the simulated series (no values of the cycle are repeated)
        T = (t[-1] - t[0]) * N / (N - 1.)
        # default amplitude is 1 and phase is 0
        A, P = 1., 0
        try:
            A, T, P = ATP
        except (TypeError, ValueError):
            try:
                A, T = ATP
            except (TypeError, ValueError):
                # default period is 1 more than the length of the simulated series
                # (no values of the cycle are repeated)
                A = ATP[0]
        # print(A, T, P)
        # print(t[1] - t[0])
        y += A * np.sin(2 * np.pi * (t - P) / T)
    if sigma:
        y += np.random.normal(0.0, float(sigma), N)
    if rw:
        y += np.random.normal(0.0, float(rw), N).cumsum()
    if irw:
        y += np.random.normal(0.0, float(irw), N).cumsum().cumsum()
    if rrw:
        y += np.random.normal(0.0, float(rrw), N).cumsum().cumsum().cumsum()
    return pd.Series(y, index=t)


def normalize_symbols(symbols, *args, **kwargs):
    """Coerce into a list of uppercase strings like "GOOG", "$SPX, "XOM"

    Flattens nested lists in `symbols` and converts all list elements to strings

    Arguments:
      symbols (str or list of str): list of market ticker symbols to normalize
        If `symbols` is a str a get_symbols_from_list() call is used to retrieve the list of symbols
      postrprocess (func): function to apply to strings after they've been stripped
        default = str.upper

    FIXME:
      - list(set(list(symbols))) and `args` separately so symbols may be duplicated in symbols and args
      - `postprocess` should be a method to facilitate monkey-patching

    Returns:
      list of str: list of cananical ticker symbol strings (typically after .upper().strip())

    Examples:
      >> normalize_symbols("Goog,AAPL")
      ['GOOG', 'AAPL']
      >> normalize_symbols("  $SPX   ", " aaPL ")
      ['$SPX', 'AAPL']
      >> normalize_symbols("  $SPX   ", " aaPL ", postprocess=str)
      ['$SPX', 'aaPL']
      >> normalize_symbols(["$SPX", ["GOOG", "AAPL"]])
      ['GOOG', 'AAPL', '$SPX']
      >> normalize_symbols("$spy", ["GOOGL", "Apple"], postprocess=str)
      ['$spy', 'GOOGL', 'Apple']
    """
    postprocess = kwargs.get('postprocess', None) or str.upper
    if (      (hasattr(symbols, '__iter__') and not any(symbols))
        or (isinstance(symbols, (list, tuple, Mapping)) and (not symbols or not any(symbols)))):
        return []
    args = normalize_symbols(args, postprocess=postprocess)
    if isinstance(symbols, basestring):
        try:
            return list(set(get_symbols_from_list(symbols))) + args
        except:
            return [postprocess(s.strip()) for s in symbols.split(',')] + args
    else:
        ans = []
        for sym in list(symbols):
            ans += normalize_symbols(sym, postprocess=postprocess)
        return list(set(ans))


def series_bollinger(series, window=20, sigma=1., plot=False):
    mean = pd.rolling_mean(series, window=window)
    std = pd.rolling_std(series, window=window)
    df = pd.DataFrame({'value': series, 'mean': mean, 'upper': mean + sigma * std, 'lower': mean - sigma * std})
    bollinger_values = (series - pd.rolling_mean(series, window=window)) / (pd.rolling_std(series, window=window))
    if plot:
        df.plot()
        pd.DataFrame({'bollinger': bollinger_values}).plot()
        plt.show()
    return bollinger_values


def frame_bollinger(df, window=20, sigma=1., plot=False):
    bol = pd.DataFrame()
    for col in df.columns:
        bol[col] = series_bollinger(df[col], plot=False)
    return bol


def double_sinc(T_0=120, T_N=240, T_s=0.01, A=[1, .9], sigma=0.01, T_cyc=10, N_cyc=[3, 2], verbosity=0):
    # T0, TN, A, sigma = np.array(T0), np.array(TN), np.array(A), np.array(sigma)
    N = int(T_N / T_s)
    t = np.arange(0, T_N, T_s)
    # t_mid = 0.5 * (t[-1] + t[0])
    e = sigma * np.random.randn(N)
    x = A[0] * np.sinc(((t - T_0) * N_cyc[0] * 2 / T_cyc) % T_cyc) * np.sinc((t - T_0) * N_cyc[1] * 2 / t[-1])
    y = x + e
    df = pd.DataFrame({'x': x, 'y': y}, index=t)
    if verbosity > 0:
        df.plot()
        plt.show(block=False)
    return df


def sinc_signals(T0=[60, 120], TN=[240, 160], A=[1, .9], sigma=[.03, .02], T_cyc=10, Ts=0.01):
    T0, TN, A, sigma = np.array(T0), np.array(TN), np.array(A), np.array(sigma)
    N1 = int(TN[0] / Ts)
    N2 = int(TN[1] / Ts)
    i1 = np.arange(0, N1)
    i2 = np.arange(0, N2)
    t1 = T0[0] + i1 * Ts
    t2 = t1[i2 + int((T0[1] - T0[0]) / Ts)]
    e1 = sigma[0] * np.random.randn(N1)
    e2 = sigma[1] * np.random.randn(N2)
    signal = A[0] * np.sinc((t1[i1] * 5. / T_cyc) % T_cyc) * np.sinc((t1[i1]) * 4 / t1[-1])
    x1 = signal + e1
    x2 = signal[i2 + int((T0[1] - T0[0]) / Ts)] + e2
    df = pd.DataFrame({'signal 1': pd.Series(x1, index=t1), 'signal 2': pd.Series(x2, index=t2)})
    df.plot()
    plt.show(block=False)
    return df


def smooth(x, window_len=11, window='hanning', fill='reflect'):
    """smooth the data using a window with requested size.

    Convolve a normalized window with the signal.

    input:
        x: signal to be smoothed
        window_len: the width of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
        fill: 'reflect' means that the signal is reflected onto both ends before filtering

    output:
        the smoothed signal

    example:

    t = linspace(-2, 2, 0.1)
    x = sin(t) + 0.1 * randn(len(t))
    y = smooth(x)

    import seaborn
    pd.DataFrame({'x': x, 'y': y}, index=t).plot()

    SEE ALSO:
      numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman
      numpy.convolve
      scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this:  instead of just y.

    References:
      http://wiki.scipy.org/Cookbook/SignalSmooth
    """

    # force window_len to be an odd integer so it can be symmetrically applied
    window_len = int(window_len)
    window_len += int(not (window_len % 2))
    half_len = (window_len - 1) / 2

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("The window arg ({}) should be 'flat', 'hanning', 'hamming', 'bartlett', or 'blackman'"
                         .format(window))

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]

    window = window.strip().lower()
    if window is None or window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y[half_len + 1:-half_len]


def estimate_shift(x, y, smoother=None, w=None, index_and_value=False, ignore_edge=1/3.,
                   method='valid'):
    """Estimate the time shift between two signals based on their cross correlation

    Arguements:
      smoother:  Smoothing function applied to correlation values before finding peak
      w:         Window. Sequence of values between 0 and 1 for wind centered on 0-shift
                 to weight correlation by before finding peak. Zero-padded to match width of
                 larger of x and y. Default = hanning(max(len(x, y)))

    Returns:
      int: number to subtract from an x index to compute a corresponding y index
    >>> x, y = np.asarray(np.matrix([[0.5, 0.01], [0.01, 1.0]]) * np.random.randn(50,2).T)
    >>> x[:30-8] = y[8:30]

    >> estimate_shift(x, y, 'full')
    -8
    >> estimate_shift(x, y, 'valid')
    -8
    >> estimate_shift(y, x, 'full') in [8, 9]
    True
    >> estimate_shift(y, x, 'full') in [8, 9]
    True
    >> estimate_shift(y, x, 'full') in [8, 9]
    True
    """
    return NotImplementedError("On Line 965, FIXME: TypeError: object of type 'NoneType' has no len()")
    method = method or 'valid'
    try:
        x = x.dropna()
        x = x.values
    except:
        pass
    try:
        y = y.dropna()
        y = y.values
    except:
        pass

    if len(x) < len(y):
        swap, x, y = -1, y, x
    else:
        swap = +1

    Nx, Ny = len(x), len(y)
    if ignore_edge > 0:
        yi0 = int(max(Ny * ignore_edge, 1))
        yi1 = max(Ny - yi0 - 1, 0)
        # ignore a large portion of the data in the shorter vector
        y = y[yi0:yi1]

    x, y = x - x.mean(), y - y.mean()
    x, y = x / x.std(),  y / y.std()

    c = np.correlate(x, y, mode=method)
    print(len(x))
    print(len(y))
    print(len(w))
    print(len(c))
    if w is not None:
        wc = int(np.ceil(len(w) / 2.)) - 1
        cc = int(np.ceil(len(c) / 2.)) - 1
        w0 = cc - wc
        print(w0)
        if w0 > 0:
            c[:w0], c[-w0:] = 0, 0
            c[w0:-w0] = w[:len(c[w0:-w0])] * c[w0:-w0]
        elif w0 == 0:
            if len(w) < len(c):
                w = np.append(w, 0)
            c = c * w[:len(c)]
        elif w0 < 0:
            w0 = abs(w0)
            w = w[w0:-w0]
            c[w0:-w0] = w[:len(c[w0:-w0])] * c[w0:-w0]
    try:
        c = smoother(c)
    except:
        pass

    offset = imax = c.argmax()
    offset = offset - yi0
    if method == 'full':
        offset = imax - Nx + 1
    # elif method == 'valid':
    #     offset = imax - yi0
    elif method == 'same':
        raise NotImplementedError("Unsure what index value to report for a correlation maximum at i = {}"
                                  .format(imax))
    offset *= swap

    if index_and_value:
        return offset, c[imax]
    else:
        return offset
estimate_offset = estimate_shift


def fuzzy_index_match(possiblities, label, **kwargs):
    """Find the closest matching column label, key, or integer indexed value

    Returns:
      type(label): sequence of immutable objects corresponding to best matches to each object in label
              if label is an int returns the object (value) in the list of possibilities at that index
              if label is a str returns the closest str match in possibilities

    >>> from collections import OrderedDict as odict
    >>> fuzzy_index_match(pd.DataFrame(pd.np.random.randn(9,4), columns=list('ABCD'), index=range(9)), 'b')
    'B'
    >>> fuzzy_index_match(odict(zip('12345','ABCDE')), 'r2d2')
    '2'
    >>> fuzzy_index_match(odict(zip('12345','ABCDE')), 1)
    '2'
    >>> fuzzy_index_match(odict(zip('12345','ABCDE')), -1)
    '5'
    >>> fuzzy_index_match(odict(zip(range(4),'FOUR')), -4)
    0
    """
    possibilities = list(possiblities)
    if isinstance(label, basestring):
        return fuzzy_get(possibilities, label, **kwargs)
    if isinstance(label, int):
        return possibilities[label]
    if isinstance(label, list):
        return [fuzzy_get(possibilities, lbl) for lbl in label]


def get_column_labels(obj):
    """Retrieve the column labels/keys from any DataFrame or QuerySet-like table object

    >>> from collections import OrderedDict
    >>> get_column_labels(OrderedDict(zip('ABC', pd.np.arange(12).reshape((3,4)))))
    ['A', 'B', 'C']
    """
    if not isinstance(obj, (list, tuple, pd.np.ndarray)):
        try:
            labels = [f.name for f in obj.model._meta.fields]
        except:
            try:
                labels = obj.keys()
            except:
                try:
                    labels = dir(obj)
                except:
                    labels = None
    elif all(isinstance(heading, basestring) for heading in obj[0]):
        labels = list(obj[0])
        # if obj isn't a reference to a mutable (dict, DataFrame, list, etc), this won't work
        del obj[0]
    return labels


def make_dataframe(obj, columns=None, exclude=None, limit=1e8):
    """Coerce an iterable, queryset, list or rows, dict of columns, etc into a Pandas DataFrame"""
    try:
        obj = obj.objects.all()[:limit]
    except:
        pass
    if isinstance(obj, (pd.Series, list, tuple)):
        return make_dataframe(pd.DataFrame(obj), columns, exclude, limit)
    # if the obj is a named tuple, DataFrame, dict of columns, django QuerySet, sql alchemy query result
    # retrieve the "include"d field/column names from its keys/fields/attributes
    if columns is None:
        columns = get_column_labels(obj)
    if exclude is not None and columns is not None and columns and exclude:
        columns = [i for i in columns if i not in exclude]
    try:
        return pd.DataFrame(list(obj.values(*columns)[:limit]))
    except:
        pass
    try:
        return pd.DataFrame(obj)[fuzzy_get(obj, columns)]
    except:
        pass
    return pd.DataFrame(obj)


def hist(table, field=-1, class_column=None,
         title='', verbosity=2, **kwargs):
    """Plot discrete PDFs

    >>> df = pd.DataFrame(pd.np.random.randn(99,3), columns=list('ABC'))
    >>> df['Class'] = pd.np.array((pd.np.matrix([1,1,1])*pd.np.matrix(df).T).T > 0)
    >>> len(hist(df, verbosity=0, class_column='Class'))
    3
    """
    field = fuzzy_index_match(table, field)
    if not isinstance(table, (pd.DataFrame, basestring)):
        try:
            table = make_dataframe(table.objects.filter(**{field + '__isnull': False}))
        except:
            table = table
    # labels = get_column_labels(table)
    try:
        table = table[pd.notnull(table[field])]
    except:
        pass

    series_labels = []
    if class_column is not None:
        series_labels = sorted(set(table[class_column]))
    labels = [str(c) for c in series_labels] + ['all']

    default_kwargs = {
        'normed': False,
        'histtype': 'bar',
        'color': seaborn.color_palette(),
        'label': labels,
        'log': True,
        'bins': 10,
        }
    default_kwargs.update(kwargs)
    num_colors = len(default_kwargs['color'])
    num_labels = len(default_kwargs['label'])
    default_kwargs['color'] = [default_kwargs['color'][i % num_colors] for i in range(num_labels)]

    if not title:
        title = '{} vs. {}'.format(titlecase(str(field).replace('_', ' ')),
                                   titlecase(str(class_column).replace('_', ' ')))
    if verbosity > 0:
        print('Plotting histogram titled: {}'.format(title))
    if verbosity > 1:
        print('histogram configuration: {}'.format(default_kwargs))
    x = [table[(table[class_column].isnull() if pd.isnull(c) else table[class_column] == c)]
         [field].values for c in series_labels]
    x += [table[field].values]
    if not default_kwargs['normed']:
        default_kwargs['weights'] = [pd.np.ones_like(x_c) / float(len(x_c)) for x_c in x]
    elif isinstance(default_kwargs['normed'], int) and default_kwargs['normed'] < 0:
        default_kwargs['normed'] = 0

    bins = default_kwargs['bins']
    # FIXME: x log scaling doesn't work
    if False and default_kwargs['log'] and isinstance(bins, int):
        max_x = max(pd.np.max(x_c) for x_c in x)
        min_x = min(pd.np.min(x_c) for x_c in x)
        if pd.isnull(min_x) or not(min_x):
            min_x = max_x / 10.
        default_kwargs['bins'] = pd.np.logspace(min_x, max_x, bins)

    fig, ax = plt.subplots()
    ans = plt.hist(x, **default_kwargs)
    # FIXME: x log scaling doesn't work
    if False and default_kwargs['log'] and isinstance(bins, int):
        ax.set_xscale('log')
    if verbosity > 1:
        plt.legend(default_kwargs['label'])
        try:
            plt.show(block=False)
        except:
            plt.show()

    plt.title(title)
    plt.xlabel(titlecase(field.replace('_', ' ')))
    if 'weights' in default_kwargs:
        plt.ylabel('Normalized Frequency or Probability')
    elif default_kwargs['normed']:
        plt.ylabel('Normalized Count')
    else:
        plt.ylabel('Count')
    if verbosity > 2:
        plt.savefig(make_timestamp() + '--' + title.replace(' ', '-') + '.png', transparent=True)
    return ans
