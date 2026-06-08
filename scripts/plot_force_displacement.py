import pandas as pd
import matplotlib.pyplot as plt
import os
 
# === 1. Load your CSV file ===

current_dir = os.path.dirname(__file__)

file_path = os.path.join(current_dir, "..", "data", "bauer_flyx.csv")

df = pd.read_csv(file_path, skiprows=[1], thousands=",")

file_name = os.path.basename(file_path)      # "bauer_flyx.csv"
file_name = os.path.splitext(file_name)[0]   # "bauer_flyx"

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
yield_index = deviation[deviation > threshold].index[0]

yield_force = df_clean.loc[yield_index, "force"]
yield_displacement = df_clean.loc[yield_index, "displacement"]

# === 6. Plot ===
plt.figure(figsize=(8, 5))

plt.plot(df_clean["displacement"], df_clean["force"], '.', markersize=2)

plt.xlabel("Displacement (mm)")
plt.ylabel("Force (N)")
plt.title(f"Force vs Displacement of {file_name}")
plt.grid()

#plt.text(0.05, 0.9, f"k ≈ {k:.2f} N/mm", transform=plt.gca().transAxes)

# Mark UTS
plt.plot(uts_displacement, uts_force, 'ro')
#plt.text(uts_displacement, uts_force, f" UTS\n({uts_force:.0f} N)", color='red')

# Mark Yield Point
plt.plot(yield_displacement, yield_force, 'go')
#plt.text(yield_displacement, yield_force, f" Yield\n({yield_force:.0f} N)", color='green')


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
plt.text(
    x_pos, y_start - line_spacing,
    f"Yield ≈ {yield_force:.0f} N",
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


plt.savefig(f"results/{file_name}_force_displacement.png", dpi=300)

plt.tight_layout()
plt.show()

import numpy as np

x = df_clean["displacement"]
y = df_clean["force"]

k, b = np.polyfit(x, y, 1)