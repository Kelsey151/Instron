import pandas as pd
import matplotlib.pyplot as plt
import os
 
# === 1. Load your CSV file ===

current_dir = os.path.dirname(__file__)

# Change file name to match your raw data file in the "data" folder
file_path = os.path.join(current_dir, "..", "data", "Specimen_RawData_1.csv")

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

# === Split into loading and unloading branches ===
peak_idx = df_clean["force"].idxmax()

loading = df_clean.loc[:peak_idx].copy().reset_index(drop=True)
unloading = df_clean.loc[peak_idx:].copy().reset_index(drop=True)

# === 5.5 Trim data after failure (keep only up to UTS) ===
# failure_cutoff_index = df_clean["force"].idxmax()
# df_clean = df_clean.loc[:failure_cutoff_index].copy().reset_index(drop=True)

import numpy as np

# === Loading stiffness ===
# Use lower-force portion of loading branch
loading_linear = loading[loading["force"] < 0.4 * loading["force"].max()]

x_load = loading_linear["displacement"]
y_load = loading_linear["force"]

k_load, b_load = np.polyfit(x_load, y_load, 1)

# === Unloading stiffness ===
# Use upper-force portion of unloading branch
unloading_linear = unloading[unloading["force"] > 0.6 * unloading["force"].max()]

x_unload = unloading_linear["displacement"]
y_unload = unloading_linear["force"]

k_unload, b_unload = np.polyfit(x_unload, y_unload, 1)

# Use absolute value so unloading stiffness displays as positive
k_unload = abs(k_unload)

# === Calculate UTS (max force) ===
uts_force = loading["force"].max()

uts_index = loading["force"].idxmax()
uts_displacement = loading.loc[uts_index, "displacement"]

# === 6. Plot ===
plt.figure(figsize=(8, 5))

# Loading branch
plt.plot(loading["displacement"], loading["force"], '.', markersize=2, label="Loading")

# Unloading branch
plt.plot(unloading["displacement"], unloading["force"], '.', markersize=2, label="Unloading")

plt.xlabel("Displacement (mm)")
plt.ylabel("Force (N)")
plt.title(f"Force vs Displacement of {file_name}")
plt.grid()

#plt.text(0.05, 0.9, f"k ≈ {k:.2f} N/mm", transform=plt.gca().transAxes)

# Mark UTS
plt.plot(uts_displacement, uts_force, 'ro')
#plt.text(uts_displacement, uts_force, f" UTS\n({uts_force:.0f} N)", color='red')

x_pos = 0.05
y_start = 0.88
line_spacing = 0.06


plt.text(
    x_pos, y_start,
    f"Loading k ≈ {k_load:.2f} N/mm",
    transform=plt.gca().transAxes,
    color='blue',
    verticalalignment='top'
)

plt.text(
    x_pos, y_start - line_spacing,
    f"Unloading k ≈ {k_unload:.2f} N/mm",
    transform=plt.gca().transAxes,
    color='orange',
    verticalalignment='top'
)

plt.text(
    x_pos, y_start - 2 * line_spacing,
    f"UTS ≈ {uts_force:.0f} N",
    transform=plt.gca().transAxes,
    color='red',
    verticalalignment='top'
)


plt.savefig(f"results/{file_name}_force_displacement.png", dpi=300)

plt.tight_layout()
plt.show()