import matplotlib.pyplot as plt
import project_paths as pp


def plot_empirical_vs_simulated_with_ci(
    edu,
    moments_sim,
    var_labels=None,
    var_scales=None,
    ylims=None,
    out_subfolder=None,
    figsize=(4, 4),
    dpi=100,
):
    if var_labels is None:
        var_labels = {}
    if var_scales is None:
        var_scales = {}
    if ylims is None:
        ylims = {}

    plot_vars = [
        v for v in edu.columns
        if v != "age" and v in moments_sim.columns
    ]

    if out_subfolder is None:
        out_dir = pp.SIM_PLOTS_DIR
    else:
        out_dir = pp.SIM_PLOTS_DIR / out_subfolder

    out_dir.mkdir(parents=True, exist_ok=True)

    for var in plot_vars:
        pretty = var_labels.get(var, var)
        scale = var_scales.get(var, 1.0)
        ymin, ymax = ylims.get(var, (None, None))

        x_emp = edu["age"]
        y_emp = edu[var] * scale

        x_sim = moments_sim["age"]
        y_sim = moments_sim[var] * scale

        low_col = f"{var}_lower"
        high_col = f"{var}_upper"
        has_ci = low_col in moments_sim.columns and high_col in moments_sim.columns

        if has_ci:
            y_low = moments_sim[low_col] * scale
            y_high = moments_sim[high_col] * scale

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(x_emp, y_emp, "-", label="Empirical")
        ax.plot(x_sim, y_sim, "--", label="Simulated")

        if has_ci:
            ax.fill_between(
                x_sim,
                y_low,
                y_high,
                alpha=0.2,
                label="95% CI",
            )

        ax.set_title(pretty)
        ax.set_xlabel("Age")
        ax.grid(True)
        ax.set_ylim(ymin, ymax)
        ax.legend()

        plt.tight_layout()

        out_dir = pp.SIM_PLOTS_DIR

        if out_subfolder is not None:
            out_dir = out_dir / out_subfolder

        out_dir.mkdir(parents=True, exist_ok=True)

        save_path = out_dir / f"{var}_over_age.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        print(f"Saved {save_path}")

        # plt.show()