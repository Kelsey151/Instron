import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# User settings
# =========================
# Folder containing taco CSV files
DATA_FOLDER = os.path.join("data", "taco")
# Folder where the combined plot will be saved
RESULTS_FOLDER = os.path.join("results", "taco")
# Name of the saved combined figure
OUTPUT_NAME = "taco_combined_overlay.png"
# Whether to draw a dashed linear stiffness fit for each curve
OVERLAY_STIFFNESS_LINES = False
# Force threshold for monotonic taco tests (used for stiffness fit region)
LINEAR_FORCE_THRESHOLD = 800


# =========================
# Load Instron CSV robustly
# =========================
def load_instron_csv(file_path):
    """
    Load an Instron CSV even if there is metadata/garbage above the real header.
    Looks for the row containing 'Extension' and 'Load', then reads from there.
    """
    with open(file_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    header_row = None
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if "extension" in line_lower and "load" in line_lower:
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"Could not find a header row containing 'Extension' and 'Load' in {Path(file_path).name}.")

    df = pd.read_csv(file_path, skiprows=header_row, thousands=",")
    df.columns = df.columns.str.strip()

    # Drop empty/unnamed columns if present
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]

    # Drop units row if first row contains values like (mm), (N)
    if len(df) > 0 and "Extension" in df.columns and "Load" in df.columns:
        first_ext = str(df.iloc[0]["Extension"]).strip()
        first_load = str(df.iloc[0]["Load"]).strip()
        if first_ext.startswith("(") or first_load.startswith("("):
            df = df.iloc[1:].reset_index(drop=True)

    if "Extension" not in df.columns or "Load" not in df.columns:
        raise ValueError(f"{Path(file_path).name}: expected columns 'Extension' and 'Load', found {list(df.columns)}")

    return df[["Extension", "Load"]]


# =========================
# Prepare data
# =========================
def prepare_curve(file_path):
    df = load_instron_csv(file_path)

    displacement = pd.to_numeric(df["Extension"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    force = pd.to_numeric(df["Load"].astype(str).str.replace(",", "", regex=False), errors="coerce")

    df_clean = pd.DataFrame({
        "displacement": displacement,
        "force": force
    }).dropna().reset_index(drop=True)

    if df_clean.empty:
        raise ValueError("No valid numeric data after cleaning.")

    # Zero to the first point
    df_clean["displacement"] -= df_clean["displacement"].iloc[0]
    df_clean["force"] -= df_clean["force"].iloc[0]

    # Trim after UTS (matches your taco workflow)
    failure_cutoff_index = df_clean["force"].idxmax()
    df_clean = df_clean.loc[:failure_cutoff_index].copy().reset_index(drop=True)

    if len(df_clean) < 2:
        raise ValueError("Not enough data points after trimming.")

    return df_clean


# =========================
# Stiffness fit (optional)
# =========================
def fit_stiffness(df_clean):
    linear_region = df_clean[df_clean["force"] < LINEAR_FORCE_THRESHOLD].copy()

    if len(linear_region) < 2:
        # Fallback: use first 30% of points if threshold gives too few values
        n = max(2, int(0.3 * len(df_clean)))
        linear_region = df_clean.iloc[:n].copy()

    x = linear_region["displacement"]
    y = linear_region["force"]

    k, b = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = k * x_fit + b

    return k, b, x_fit, y_fit


# =========================
# Main: overlay all curves
# =========================
def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    data_dir = (repo_root / "data" / "taco").resolve()
    results_dir = (repo_root / "results" / "taco").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    plt.figure(figsize=(9, 6))

    summary_lines = []

    for csv_file in csv_files:
        try:
            df_clean = prepare_curve(csv_file)
            label = csv_file.stem

            # Plot curve
            plt.plot(
                df_clean["displacement"],
                df_clean["force"],
                linewidth=2,
                label=label
            )

            # Optional stiffness overlay
            if OVERLAY_STIFFNESS_LINES:
                k, b, x_fit, y_fit = fit_stiffness(df_clean)
                plt.plot(x_fit, y_fit, linestyle="--", linewidth=1.2)
                summary_lines.append(f"{label}: k ≈ {k:.2f} N/mm")

        except Exception as e:
            summary_lines.append(f"Skipped {csv_file.stem}: {e}")

    plt.xlabel("Displacement (mm)")
    plt.ylabel("Force (N)")
    plt.title("Combined Force vs Displacement Overlay")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()

    save_path = results_dir / OUTPUT_NAME
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved combined overlay to: {save_path}")
    if summary_lines:
        print("\nSummary:")
        for line in summary_lines:
            print(" -", line)


if __name__ == "__main__":
    main()
