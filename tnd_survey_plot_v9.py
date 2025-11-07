import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ===============================
#  TND Survey Plot - v9
# ===============================

# Input result file from tnd_model_v9.py
input_file = "results/results_tnd_v9.csv"

# Output figure
output_plot = "results/tnd_survey_plot_v9.png"

# Read results
df = pd.read_csv(input_file)

# Ensure consistent column names
df.columns = [c.strip() for c in df.columns]

# Try to detect columns
depth_col = [c for c in df.columns if "depth" in c.lower()][0]
tension_col = [c for c in df.columns if "tension" in c.lower() and "rot" not in c.lower()][0]
torque_col = [c for c in df.columns if "torque" in c.lower() and "rot" not in c.lower()][0]

# Check optional rotating/sliding
tension_rot_col = next((c for c in df.columns if "tension_rot" in c.lower()), None)
torque_rot_col = next((c for c in df.columns if "torque_rot" in c.lower()), None)

# Depth (ft)
depth = df[depth_col]

# Prepare figure
plt.figure(figsize=(11, 6))
plt.suptitle("Torque & Drag vs Measured Depth (v9 Model)", fontsize=14, fontweight="bold")

# --- Left Plot: Tension ---
plt.subplot(1, 2, 1)
plt.plot(df[tension_col], depth, label="Sliding", color="tab:red", linewidth=2)
if tension_rot_col:
    plt.plot(df[tension_rot_col], depth, "--", label="Rotating", color="tab:blue", linewidth=2)

plt.gca().invert_yaxis()
plt.xlabel("Tension (lbf)")
plt.ylabel("Measured Depth (ft)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.title("Tension Profile")

# --- Right Plot: Torque ---
plt.subplot(1, 2, 2)
plt.plot(df[torque_col], depth, label="Sliding", color="tab:red", linewidth=2)
if torque_rot_col:
    plt.plot(df[torque_rot_col], depth, "--", label="Rotating", color="tab:blue", linewidth=2)

plt.gca().invert_yaxis()
plt.xlabel("Torque (ft-lbf)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.title("Torque Profile")

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot
os.makedirs("results", exist_ok=True)
plt.savefig(output_plot, dpi=300)
plt.close()

print(f"[✓] Torque & Drag plot saved: {output_plot}")
