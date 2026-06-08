import pandas as pd
import matplotlib.pyplot as plt
import os
 
# === 1. Load your CSV file ===

current_dir = os.path.dirname(__file__)

# Change file name to match your raw data file in the "data" folder
file_path = os.path.join(current_dir, "..", "data", "taco_test_000.csv")

df = pd.read_csv(file_path, thousands=",")

file_name = os.path.basename(file_path)
file_name = os.path.splitext(file_name)[0]

df.columns = df.columns.str.strip()

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

# === 5.5 Trim data after failure (keep only up to UTS) ===
failure_cutoff_index = df_clean["force"].idxmax()
df_clean = df_clean.loc[:failure_cutoff_index].copy().reset_index(drop=True)

import numpy as np

linear_region = df_clean[df_clean["force"] < 800]

x = linear_region["displacement"]
y = linear_region["force"]

k, b = np.polyfit(x, y, 1)

# === 6. Calculate UTS (max force) ===
uts_force = df_clean["force"].max()

# Get displacement at UTS
uts_index = df_clean["force"].idxmax()
uts_displacement = df_clean.loc[uts_index, "displacement"]

# === 6. Plot ===
plt.figure(figsize=(8, 5))

plt.plot(df_clean["displacement"], df_clean["force"], '.', markersize=2)

plt.xlabel("Displacement (mm)")
plt.ylabel("Force (N)")
plt.title(f"Force vs Displacement of {file_name}")
plt.grid()

# Mark UTS
plt.plot(uts_displacement, uts_force, 'ro')

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

# UTS (red)
plt.text(
    x_pos, y_start - 1 * line_spacing,
    f"UTS ≈ {uts_force:.0f} N",
    transform=plt.gca().transAxes,
    color='red',
    verticalalignment='top'
)

plt.savefig(f"results/{file_name}_force_displacement.png", dpi=300)

plt.tight_layout()
plt.show()