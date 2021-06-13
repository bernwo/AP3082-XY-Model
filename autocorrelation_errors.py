import numpy as np
from scipy.optimize.minpack import curve_fit

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

def func(x,a,b,c):
    """
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
    return a*np.exp(-x/b)+c

def get_tau(autocorrelation):
    """
    Parameters:
    -----------
    autocorrelation: np.darray
        autocorrelation function of a given observable

    Return:
    -------
    params: np.darray
        fitting parameters of autocorrelation onto an exponential decay
    """
    N = len(autocorrelation)//2
    x = np.arange(N)
    y = np.nan_to_num(autocorrelation[:N])
    params, errors = curve_fit(func,x, y)
    return params

def get_error_observable(observable):
    """
    Parameters:
    -----------
    observable: np.ndarray
        data of an observable w.r.t. time

    Return:
    --------
    sigma: float
        error associated to a given observable
    """
    observable=np.nan_to_num(observable)
    tau = get_tau(np.nan_to_num(get_autocorrelation_function(observable)))[1]
    N = len(observable)
    sigma = np.sqrt(2*tau/N)*np.sqrt(np.mean(observable**2)-np.mean(observable)**2)
    return sigma
