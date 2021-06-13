import numpy as np
from scipy.optimize.minpack import curve_fit

"""
    Data blocking error calculation
"""

def autocorrelation_function(observable, time):
    """
    Parameters:
    -----------
    observable: np.ndarray
        data of an observable w.r.t. time
    time: float
        time at which autocorrelation function is computed

    Return:
    autocorrelation: float
        autocorrelation function at a given time
    --------
    """
    #Number of time steps
    N = len(observable)

    #Long time corrections tend to mess up data, this next line ignores corrections after half time
    #observable[N//:]=0

    A_nt = observable[time:N-time:time]
    A_n = observable[:(N-time)//time]
    A_nn = observable[:N-time]

    #Make sure that A_nt and A_n have the same length
    if len(A_nt)!=len(A_n):
        A_n = observable[:(N-time)//time-1]

    #Here the term sum_n A_{n} A_{n+t} was implemented as np.sum(A_nt * A_n)
    #Other terms like sum_n A_n were implemented as np.sum(A_nn)
    #Note that len(A_nn) > len(A_n)
    sigma_A_n = np.sqrt((N-time)*np.sum(A_nn**2)-(np.sum(A_nn)**2))
    sigma_A_nt = np.sqrt((N-time)*np.sum(A_nt**2)-(np.sum(A_nt)**2))
    autocorrelation = ((N-time)*np.sum(A_n*A_nt)-np.sum(A_n)*np.sum(A_nt))/(sigma_A_n*sigma_A_nt)

    return np.abs(autocorrelation)

def get_autocorrelation_function(observable):
    """
    Parameters:
    -----------
    observable: np.ndarray
        data of an observable w.r.t. time

    Return:
    af: np.darray
        autocorrelation function of an observable over all time
    --------
    """
    number_steps = len(observable)-1
    af = np.zeros(number_steps)
    for time in range(1,number_steps):
        af[time]=autocorrelation_function(observable, time)

    return af[1:]


def split_padded(a,n):
    """
    Taken from: https://stackoverflow.com/questions/9922395/python-numpy-split-array-into-unequal-subarrays
    """
    padding = (-len(a))%n
    return np.split(np.concatenate((a,np.zeros(padding))),n)

def func_block(x,b):
    """
    Function used to fit the decaying exponential of the autocorrelation func

    Parameters:
    -----------
        x: float

        a: float
            magnitude
        b: float
            characteristic distance
        c: float
            shift from zero

    Return:
    --------
        _: float
            function of exponential decay evaluated at x,a,b,c.
    """
    return np.exp(-x/b)


def get_tau_block(autocorrelation):
    """
    Obtain characteristic length of the autocorrelation func

    Parameters:
    -----------
        autocorrelation: np.darray
            autocorrelation function of a given observable

    Return:
    -------
        params: np.darray
            fitting parameters of autocorrelation onto an exponential decay
    """
    N = len(autocorrelation)
    x = np.arange(N)
    y = np.nan_to_num(autocorrelation[:N])
    try:
        params, errors = curve_fit(func_block, x, y)
    except:
        params, errors = 0, 0
    perr = np.sqrt(np.diag(errors))
    return params, perr

def get_error_block(obs,T_i,T_f,size=10,tau=None,tau_error=None):
    """
    Use tau to calculate statistical error of a data series using the block data method.
    Receives the full observable and retrieves an equal (tau=1) or shorter list where
    data was separated into uncorrelated blocks based on the autocorrelation function.

    Parameters:
    -----------
        obs: np.ndarray
            data of an observable w.r.t. time
        T_i: float
            Initial temperature
        T_f: float
            Final temperature
        dT: float
            Temperature step
        size: int
            Size of the lattice
        tau: float
            Correlation time. If not specified it is calculated for the data set
        tau_error: float
            Error of the correlation time. Calculated if not specified
    Return:
    --------
        temps_m: nd.array
            Temperatures for blocked data
        blocked_m: nd.array
            Uncorrelated version of the given observable
        error_m: nd.array
            Error calculated for each block of data
        tau: float
            Correlation time
        tau_error: float
            Error associated to the correlation time
    """
    if tau == None:
        tau, tau_error = get_tau_block(get_autocorrelation_function(obs)[:size])
        tau = int(np.round(tau))

    if tau > 1 and tau < len(obs):
        blocked_m = split_padded(obs,tau)
        error_m = np.std(np.array(blocked_m),axis=1)
        blocked_m = np.mean(blocked_m,axis=1)

    else:
        blocked_m = obs
        error_m = 0
        tau = 1

    len_data = len(blocked_m)
    temps_m = np.linspace(T_i, T_f, num=len_data)

    return temps_m, blocked_m, error_m, tau, tau_error
