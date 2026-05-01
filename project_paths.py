from pathlib import Path

# Root folder of the whole project
DIR = Path(__file__).resolve().parent

DATA_DIR = DIR / "data"
MOMENTS_DIR = DATA_DIR / "momenter"

RESULTS_DIR = DIR / "results"
FIRST_STAGE_RESULTS_DIR = RESULTS_DIR / "first_stage estimation"
STRUCTURAL_RESULTS_DIR = RESULTS_DIR / "structural estimation"
COUNTERFACTUALS_RESULTS_DIR = RESULTS_DIR / "counterfactuals"
CF_EXP_DIR = COUNTERFACTUALS_RESULTS_DIR / "expected"
CF_UNEXP_DIR = COUNTERFACTUALS_RESULTS_DIR / "unexpected"
SIM_RESULTS_DIR = RESULTS_DIR / "simulation"

FIRST_STEP_DIR = DIR / "first_step"

PLOTS_DIR = DIR / RESULTS_DIR / "plots"
CF_PLOTS_DIR = PLOTS_DIR / "counterfactuals"
CF_UNEXP_PLOTS_DIR = CF_PLOTS_DIR / "unexpected"
CF_EXP_PLOTS_DIR = CF_PLOTS_DIR / "expected"
SIM_PLOTS_DIR = PLOTS_DIR / "simulation"

EDU1_DIR = DIR / "EDU1"
EDU2_DIR = DIR / "EDU2"
EDU3_DIR = DIR / "EDU3"

MODEL_FUNCTIONS_DIR = DIR / "model_functions_initial"
MODEL_FUNCTIONS_COUNTERFACTUALS_DIR = DIR / "model_functions_counter"