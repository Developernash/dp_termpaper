#system imports
from pathlib import Path
from typing import Tuple
from typing import Dict
import os
import sys

#fundamental imports
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns


#Project imports
from model_functions_initial.utility import utility_functions
from model_functions_initial.final_period_utility import final_period_utility
from model_functions_initial.budget import budget_dcegm_initial
from model_functions_initial.state_space_functions import create_state_space_function_dict
from model_functions_initial.compute_moments import compute_simulation_moments
from model_functions_initial.compute_moments import compute_simulation_moments_with_ci
# from model_functions_initial.estimation import estimate_msm

# Project counterfactuals
from model_functions_counter.compute_counterfac import compute_counterfactual_diff
from model_functions_counter.compute_counterfac import compute_diff_by_edu
from model_functions_counter.compute_counterfac import plot_metrics_individual
from model_functions_counter.compute_counterfac import plot_counterfactual_diff
from model_functions_counter.compute_counterfac import plot_cf_diff_separate

from model_functions_counter.budget_counter     import budget_dcegm_counter_oap

from plots.plots import plot_empirical_vs_simulated_with_ci

from first_step.load_params import load_params_txt
from first_step.mortality import prob_survival

#DC-EGM imports
# import dcegm
from dcegm.simulation.sim_utils import create_simulation_df

# #old dcegm
# from dcegm.interface                  import validate_exogenous_processes
# from dcegm.sim_interface              import get_sol_and_sim_func_for_model

# Trying
from dcegm.pre_processing.setup_model import (
    create_model_dict,
    create_model_dict_and_save,
    load_model_dict,
    process_debug_string
)

#Statistical imports
import statsmodels.api as sm
from scipy.optimize import minimize

