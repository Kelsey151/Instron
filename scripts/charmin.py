import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
 
def load_instron_csv(file_path):
    """
    Load an Instron CSV even if there is garbage/metadata above the real header.
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
        raise ValueError("Could not find a header row containing 'Extension' and 'Load'.")

    # Read starting from the detected header row
    df = pd.read_csv(file_path, skiprows=header_row, thousands=",")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Drop unnamed/empty columns if they exist
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]

    # If the first data row is actually the units row, drop it
    if len(df) > 0:
        first_ext = str(df.iloc[0]["Extension"]).strip() if "Extension" in df.columns else ""
        first_load = str(df.iloc[0]["Load"]).strip() if "Load" in df.columns else ""

        if first_ext.startswith("(") or first_load.startswith("("):
            df = df.iloc[1:].reset_index(drop=True)

    return df

# === 1. Load your CSV file ===

current_dir = os.path.dirname(__file__)

# Change file path to match your raw data file in the "data" folder
file_path = os.path.join(current_dir, "..", "data", "charmin", "white_day0.csv")
df = load_instron_csv(file_path)

file_name = os.path.basename(file_path)
file_name = os.path.splitext(file_name)[0]

# === 3. Extract force and displacement ===

displacement = pd.to_numeric(df["Extension"].astype(str).str.replace(",", ""), errors="coerce")
force = pd.to_numeric(df["Load"].astype(str).str.replace(",", ""), errors="coerce")

# === 4. Clean data (remove NaNs) ===
df_clean = pd.DataFrame({
    "displacement": displacement,
    "force": force
}).dropna()

# === 5. Zero the data (important for Instron tests) ===
df_clean["displacement"] -= df_clean["displacement"].iloc[0]
df_clean["force"] -= df_clean["force"].iloc[0]

# === Split into loading and unloading branches ===
peak_idx = df_clean["force"].idxmax()

loading = df_clean.loc[:peak_idx].copy().reset_index(drop=True)
unloading = df_clean.loc[peak_idx:].copy().reset_index(drop=True)


## === Automatic spike detection on loading branch ===

# Smooth the loading force slightly to make slope detection more stable
loading["force_smooth"] = loading["force"].rolling(window=5, center=True, min_periods=1).mean()

# Compute local slope dF/dx
slope = np.gradient(loading["force_smooth"], loading["displacement"])

# Estimate the "normal" loading slope using the middle of the branch
mid_region = loading[
    (loading["displacement"] > 0.1 * loading["displacement"].max()) &
    (loading["displacement"] < 0.8 * loading["displacement"].max())
].copy()

mid_slope = np.gradient(mid_region["force_smooth"], mid_region["displacement"])
baseline_slope = np.median(mid_slope)

# Detect the spike near the end of the curve:
# look only near the high-displacement end, and find where slope jumps strongly
candidate_region = loading["displacement"] > 0.8 * loading["displacement"].max()

spike_candidates = loading.index[
    candidate_region &
    #edit this line to adjust sensitivity of spike detection:
    (slope > 2.5 * baseline_slope)
]

# If a spike is found, cut at the first spike point
# otherwise keep the full loading branch
if len(spike_candidates) > 0:
    cutoff_idx = spike_candidates[0]
else:
    cutoff_idx = loading.index[-1]

loading_trimmed = loading.loc[:cutoff_idx].copy().reset_index(drop=True)

# === Loading stiffness from trimmed loading curve ===
x_load = loading_trimmed["displacement"]
y_load = loading_trimmed["force"]

k_load, b_load = np.polyfit(x_load, y_load, 1)

## === 6. Plot ===
plt.figure(figsize=(8, 5))

# Full loading and unloading curves
plt.plot(
    loading["displacement"],
    loading["force"],
    '-',
    linewidth=2,
    color='tab:blue',
    label="Loading"
)

# Plot the loading stiffness fit line over the trimmed region
x_fit_line = np.linspace(x_load.min(), x_load.max(), 100)
y_fit_line = k_load * x_fit_line + b_load

plt.plot(
    x_fit_line,
    y_fit_line,
    '--',
    linewidth=2,
    color='black',
    label="Stiffness"
)

# Optionally mark automatic cutoff point
cutoff_displacement = loading_trimmed["displacement"].iloc[-1]
cutoff_force = loading_trimmed["force"].iloc[-1]
plt.plot(cutoff_displacement, cutoff_force, 'ko', markersize=5)

plt.xlabel("Displacement (mm)")
plt.ylabel("Force (N)")
plt.title(f"Force vs Displacement of {file_name}")
plt.grid()

# Text block
x_pos = 0.05
y_start = 0.88
line_spacing = 0.06

plt.text(
    x_pos, y_start,
    f"Loading k ≈ {k_load:.2f} N/mm",
    transform=plt.gca().transAxes,
    color='black',
    verticalalignment='top'
)

plt.legend()
plt.tight_layout()

plt.savefig(f"results/charmin/white_mormon/{file_name}_force_displacement.png", dpi=300)
plt.show()