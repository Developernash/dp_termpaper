import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import project_paths as pp

#=================================
#       Data preparation
#=================================

# Files by education group
files = {
    1: pp.MOMENTS_DIR / "moments_udd1.txt",
    2: pp.MOMENTS_DIR / "moments_udd2.txt",
    3: pp.MOMENTS_DIR / "moments_udd3.txt",
}

# Load data
dfs = {udd: pd.read_csv(path)
        for udd, path in files.items()}

# Standardize column names and remove _FREQ_
for udd, df in dfs.items():
    if "ALDER" in df.columns:
        df.rename(columns={"ALDER": "age"}, inplace=True)

    if "_FREQ_" in df.columns:
        df.drop(columns=["_FREQ_"], inplace=True)

#=================================
#       Wage estimation for each education group
#       OLS: log(wage) = β0 + β1·age + β2·age² + error
#=================================

# Store models and betas
models = {}
betas = {}

for udd, df in dfs.items():

    # Build regression dataset
    df["log_wage"] = np.log(df["avg_wage"])
    X = sm.add_constant(np.column_stack((df["age"], df["age"] ** 2)))

    # Estimate OLS
    model = sm.OLS(df["log_wage"], X, missing="drop").fit()

    # Extract betas
    beta0, beta1, beta2 = model.params

    models[udd] = model
    betas[udd] = (beta0, beta1, beta2)

    print(f"UDD {udd}: β0={beta0:.4f}, β1={beta1:.4f}, β2={beta2:.4f}")

    # Save parameters
    np.savetxt(pp.FIRST_STAGE_RESULTS_DIR / f"wage_params_udd{udd}.txt", [beta0, beta1, beta2])

    # Add prediction to dataframe
    df["predicted"] = np.exp(beta0 + beta1 * df["age"] + beta2 * df["age"] ** 2)