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

# === 6. Plot ===
plt.figure(figsize=(8, 5))

plt.plot(df_clean["displacement"], df_clean["force"], '.', markersize=2)

plt.xlabel("Displacement (mm)")
plt.ylabel("Force (N)")
plt.title(f"Force vs Displacement of {file_name}")
plt.grid()

plt.savefig(f"results/{file_name}_force_displacement.png", dpi=300)

plt.tight_layout()

plt.text(0.05, 0.9, f"k ≈ {k:.2f} N/mm", transform=plt.gca().transAxes)

plt.show()

import numpy as np

x = df_clean["displacement"]
y = df_clean["force"]

k, b = np.polyfit(x, y, 1)