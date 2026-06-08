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

# Change file name to match your raw data file in the "data" folder
file_path = os.path.join(current_dir, "..", "data","blade_3pt_bend", "ccm_xs1pro.csv")

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

linear_region = df_clean[df_clean["force"] < 800]

x = linear_region["displacement"]
y = linear_region["force"]

k, b = np.polyfit(x, y, 1)

# === 6. Calculate UTS (max force) ===
uts_force = df_clean["force"].max()

# Get displacement at UTS
uts_index = df_clean["force"].idxmax()
uts_displacement = df_clean.loc[uts_index, "displacement"]

# === 7. Estimate Yield Point ===

# Use same linear region you used for stiffness
linear_region = df_clean[df_clean["force"] < 800]

x_lin = linear_region["displacement"]
y_lin = linear_region["force"]

# Linear fit (you already do this)
k, b = np.polyfit(x_lin, y_lin, 1)

# Predict force using linear model
predicted_force = k * df_clean["displacement"] + b

# Compute deviation
deviation = abs(df_clean["force"] - predicted_force)

# Define threshold (tweak if needed)
threshold = 50  # N deviation

# Find first point where deviation exceeds threshold

yield_candidates = deviation[deviation > threshold]

if len(yield_candidates) == 0:
    print("No yield point found with the current threshold.")
    yield_force = None
    yield_displacement = None
else:
    yield_index = yield_candidates.index[0]
    yield_force = df_clean.loc[yield_index, "force"]
    yield_displacement = df_clean.loc[yield_index, "displacement"]

# === 6. Plot ===
plt.figure(figsize=(8, 5))

plt.plot(df_clean["displacement"], df_clean["force"], '.', markersize=2)

plt.xlabel("Displacement (mm)")
plt.ylabel("Force (N)")
plt.title(f"Force vs Displacement of {file_name}")
plt.grid()

# Mark UTS
plt.plot(uts_displacement, uts_force, 'ro')

# Mark Yield Point only if it exists
if yield_force is not None and yield_displacement is not None:
    plt.plot(yield_displacement, yield_force, 'go')

# Position (top-left)
x_pos = 0.05
y_start = 0.88
line_spacing = 0.06

# Stiffness (black)
plt.text(
    x_pos, y_start,
    f"k ≈ {k:.2f} N/mm",
    transform=plt.gca().transAxes,
    color='black',
    verticalalignment='top'
)

# Yield (green)
if yield_force is not None:
    plt.text(
        x_pos, y_start - line_spacing,
        f"Yield ≈ {yield_force:.0f} N",
        transform=plt.gca().transAxes,
        color='green',
        verticalalignment='top'
    )
else:
    plt.text(
        x_pos, y_start - line_spacing,
        "Yield not found",
        transform=plt.gca().transAxes,
        color='green',
        verticalalignment='top'
    )

# UTS (red)
plt.text(
    x_pos, y_start - 2 * line_spacing,
    f"UTS ≈ {uts_force:.0f} N",
    transform=plt.gca().transAxes,
    color='red',
    verticalalignment='top'
)

plt.savefig(f"results/blade_3pt_bend/{file_name}_force_displacement.png", dpi=300)

plt.tight_layout()
plt.show()