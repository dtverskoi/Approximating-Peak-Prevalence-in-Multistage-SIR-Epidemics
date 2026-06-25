import numpy as np
from scipy.integrate import solve_ivp
import xarray as xr
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns


def wrapper(r0_vect, k_vect, t_max, epsilon, gamma):
    """
    wrapper: Wraps all function executions into one function

    :param r0_vect: vector containing all desired r0 values
    :param k_vect: vector containing all desired k values - number of compartments
    :param t_max: max time to run ODE integration for
    :param epsilon: initial proportion of infected individuals
    :param gamma: mean time spent in each compartment

    :return: two dataframes, first contains bias of all scenarios, second contains estimates for all scenarios
    """

    # Creates empty data array to store bias and estimates
    bias = xr.DataArray(
        np.full((len(r0_vect), len(k_vect), 3), np.nan),
        coords={
            "r0": r0_vect,
            "k": k_vect,
            "Method": ["Simple", "Fully Corrected", "L1"]
        },
        dims=["r0", "k", "Method"]
    )
    estimates = xr.DataArray(
        np.full((len(r0_vect), len(k_vect), 3), np.nan),
        coords={
            "r0": r0_vect,
            "k": k_vect,
            "Method": ["Simple", "Fully Corrected", "L1"]
        },
        dims=["r0", "k", "Method"]
    )

    for r0 in r0_vect:
        for k in k_vect:
            # For each simulation scenario, solve for delta, beta, and lambda
            delta = k * gamma
            beta = r0 * gamma
            lambda_val = np.sqrt((r0 ** 2 + 1) / 2)

            # Solve ODE to find actual peak prevalence
            sol = ode_solver(k, t_max, epsilon, beta, delta)
            i_max_real = float(np.max(np.sum(sol.y[1:-1], axis=0)))

            # Explicitly solve for v_max as a function of parameters
            v_max = k - delta / beta - delta / beta * np.log(beta * k * (1 - epsilon) / delta)
            w_max = v_max / (k + 1)

            # Calculate each estimate as C * w_max
            simple_est = 2 * w_max
            fc_est = fc_est_fn(lambda_val) * w_max
            # Only calculate large lambda estimate if lambda is large
            if lambda_val > np.sqrt(2 * np.pi):
                l1_est = l1_est_fn(lambda_val) * w_max
            else:
                l1_est = np.nan

            # Calculate error for each estimate
            error_simple = (simple_est - i_max_real) / i_max_real
            error_fc = (fc_est - i_max_real) / i_max_real
            if not np.isnan(l1_est):
                error_l1 = (l1_est - i_max_real) / i_max_real
            else:
                error_l1 = np.nan

            # Store bias and estimates
            bias.loc[dict(r0 = r0, k = k)] = [error_simple, error_fc, error_l1]
            estimates.loc[dict(r0 = r0, k = k)] = [simple_est, fc_est, l1_est]

    return bias.to_dataframe(name="bias").reset_index(), estimates.to_dataframe(name="estimates").reset_index()

def sikr(t, y, beta, delta):
    """
    sikr: defines ODE for SIkR model

    :param t: time for ODE to be calculated - passed by solve_ivp
    :param y: vector containing states of compartments
    :param beta: mean infection rate
    :param delta: mean infectious time

    :return: derivatives for each compartment at time t
    """
    s = float(y[0])
    i = np.array(y[1:-1])
    r = float(y[-1])
    s_dot = - beta * s * np.sum(i) / (s + np.sum(i) + r)
    i_dot = np.empty(len(i))
    i_dot[0] = beta * s * np.sum(i) / (s + np.sum(i) + r) - delta * i[0]
    for j in range(1, len(i)):
        i_dot[j] = delta * i[j - 1] - delta * i[j]
    r_dot = delta * i[len(i) - 1]
    return np.concatenate([[s_dot], i_dot, [r_dot]])



def ode_solver(k, t_max, epsilon, beta, delta):
    """
    ode_solver: solves for ODE of SIkR model

    :param k: number of infectious compartments
    :param t_max: Max time to evaluate ODE
    :param epsilon: initial proportion of infected individuals
    :param beta: mean infection rate
    :param delta: mean infectious time

    :return: ODE trajectories for each compartment
    """
    y0 = [1 - epsilon] + [epsilon] + [0] * k

    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, 0.1)

    sol = solve_ivp(
        sikr,
        t_span,
        y0,
        t_eval=t_eval,
        args=(beta, delta),
        method='RK45',
        dense_output=True
    )
    return sol

def integrand(r, x, lambda_val):
    """
    integrand: solves for integrand 
    
    :param r: variable of integration
    :param x: variable we maximize over
    :param lambda_val: length of infectious period relative to local width of incidence peak

    :return: Value of integrand at given values
    """
    return (1 - r / lambda_val) * np.exp(-(x - r)**2 / 2)

def integral_up_to_lambda(x, lambda_val):
    """
    integral_up_to_lambda: solves integral from 0 to lambda_val

    :param x: variable we maximize over
    :param lambda_val: length of infectious period relative to local width of incidence peak

    :return: Estimated integral
    """
    result, _ = quad(integrand, 0, lambda_val, args=(x, lambda_val))
    return result

def fc_est_fn(lambda_val):
    """
    fc_est_fn: Calculates correction needed for fully-corrected estimate

    :param lambda_val: length of infectious period relative to local width of incidence peak

    :return: Scalar required for fully-corrected estimate of peak prevalence
    """

    # Calculate J_i(lambda)
    j_i_lambda = np.sqrt(2 * np.pi) * (2 * norm.cdf(lambda_val / 2) - 1)

    # Calculate J_w(lambda)
    # Minimize over negative of the integral
    j_w_lambda = -minimize_scalar(lambda x: -integral_up_to_lambda(x, lambda_val),
                                  bounds=(0, lambda_val),
                                  method='bounded').fun
    # Return ratio as C(lambda)
    return j_i_lambda / j_w_lambda


def l1_est_fn(lambda_val):
    """
    l1_est_fn: Calculates correction needed for large lambda one-step correction approximation

    :param lambda_val: length of infectious period relative to local width of incidence peak

    :return: Scalar required for large lambda one-step correction estimate of peak prevalence
    """
    # Calculates C(lambda)
    x_w_0 = np.sqrt(2 * np.log(lambda_val / np.sqrt(2 * np.pi)))
    x_w_1 = np.sqrt(2 * np.log(lambda_val / (norm.cdf(x_w_0) * np.sqrt(2 * np.pi))))
    corr = ((np.sqrt(2 * np.pi) * (2 * norm.cdf(lambda_val / 2) - 1)) /
            (np.exp(-x_w_1 ** 2 / 2) * (lambda_val - x_w_1 - (1 / lambda_val))))
    return corr

def plot_methods(data, **kwargs):
    """
    plot_methods: Helper function to create panel plot

    :param data: dataset with bias for each k and R0 value
    :param kwargs: kwargs for plotting

    :return: NONE
    """
    styles = {
        "Simple": {"color": "blue", "linestyle": "-", "marker": "o", "markerfacecolor": "none", "markersize": 4},
        "Fully Corrected": {"color": "red", "linestyle": "--", "marker": "s", "markerfacecolor": "none", "markersize": 4},
        "L1": {"color": "orange", "linestyle": (0, (3, 1)), "marker": "^", "markerfacecolor": "none", "markersize": 4},
    }

    for method, style in styles.items():
        subset = data[data["Method"] == method]
        plt.plot(subset["k"], subset["bias"], **style, label=method)

def plot_bias_results(bias):
    """
    plot_bias_results: Wrapper function to create panel plot

    :param bias: dataset with bias for each k and R0 value

    :return: NONE
    """
    g = sns.FacetGrid(
        bias,
        col="r0",
        col_wrap=6,
        sharey=True,
        ylim=(-0.3, 0.3)
    )

    g.map_dataframe(plot_methods)
    g.map(plt.axhline, y=0.1, linestyle="--", color="black", linewidth=0.8)
    g.map(plt.axhline, y=-0.1, linestyle="--", color="black", linewidth=0.8)
    g.map(plt.axhline, y=0, linestyle="-", color="black", linewidth=0.8)
    handles, labels = g.axes[0].get_legend_handles_labels()
    g.figure.legend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc="center left")
    plt.tight_layout()
    plt.show()
