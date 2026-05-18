from pathlib import Path

# Root folder of the whole project
DIR = Path(__file__).resolve().parent

DATA_DIR = DIR / "1. Data"
MOMENTS_DIR = DATA_DIR / "momenter"

RESULTS_DIR = DIR / "4. Results"
FIRST_STAGE_RESULTS_DIR = RESULTS_DIR / "first_stage estimation"
STRUCTURAL_RESULTS_DIR = RESULTS_DIR / "Structural estimation"
COUNTERFACTUALS_RESULTS_DIR = RESULTS_DIR / "Counterfactuals"
CF_EXP_DIR = COUNTERFACTUALS_RESULTS_DIR / "expected"
CF_UNEXP_DIR = COUNTERFACTUALS_RESULTS_DIR / "unexpected"
SIM_RESULTS_DIR = RESULTS_DIR / "Simulation"

FIRST_STEP_DIR = DIR / "2. First step estimation"

PLOTS_DIR = DIR / RESULTS_DIR / "plots"
CF_PLOTS_DIR = PLOTS_DIR / "counterfactuals"
CF_UNEXP_PLOTS_DIR = CF_PLOTS_DIR / "unexpected"
CF_EXP_PLOTS_DIR = CF_PLOTS_DIR / "expected"
SIM_PLOTS_DIR = PLOTS_DIR / "simulation"

MODEL_FUNCTIONS_DIR = DIR / "0. Functions"