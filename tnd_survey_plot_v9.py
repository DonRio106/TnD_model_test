import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

print("📊 Generating enhanced Torque & Drag survey plot v10...")

# =============================
# Load results file
# =============================
results_path = "results/results_tnd_v9.csv"
if not os.path.exists(results_path):
    print(f"❌ Error: Results file not found → {results_path}")
    sys.exit(1)

df = pd.read_csv(results_path)
print(f"✅ Loaded {len(df)} rows from {results_path}")

# =============================
# Identify key columns
# =============================
depth_col = next((c for c in df.columns if any(k in c.lower() for k in ["depth", "md", "measured_depth"])), None)
tension_col = next((c for c in df.columns if "tension" in c.lower() or "pickup" in c.lower()), None)
torque_col = next((c for c in df.columns if "torque" in c.lower()), None)
mu_col = next((c for c in df.columns if any(k in c.lower() for k in ["mu", "friction"])), None)
mode_col = next((c for c in df.columns if any(k in c.lower() for k in ["mode", "run_type", "operation"])), None)

if not all([depth_col, tension_col, torque_col, mu_col]):
    print("❌ Missing one or more required columns for plotting.")
    print(f"Depth: {depth_col}, Tension: {tension_col}, Torque: {torque_col}, Mu: {mu_col}")
    sys.exit(1)

# Create Mode column if missing
if mode_col is None:
    df["Mode"] = "All Runs"
    mode_col = "Mode"

# =============================
# Setup plot canvas
# =============================
unique_mus = sorted(df[mu_col].dropna().unique())
unique_modes = sorted(df[mode_col].dropna().unique())

print(f"🎯 μ values found: {unique_mus}")
print(f"🔁 Run types found: {unique_modes}")

fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
ax_tension, ax_torque = axes
colors = plt.cm.tab10.colors

# =============================
# Plot all combinations
# =============================
for i, mu in enumerate(unique_mus):
    for j, mode in enumerate(unique_modes):
        subset = df[(df[mu_col] == mu) & (df[mode_col] == mode)]
        if subset.empty:
            continue
        color = colors[i % len(colors)]
        style = ["-", "--", "-.", ":"][j % 4]
        label = f"μ={mu:.2f} ({mode})"

        # Plot tension
        ax_tension.plot(subset[tension_col], subset[depth_col], style, color=color, label=label)
        # Plot torque
        ax_torque.plot(subset[torque_col], subset[depth_col], style, color=color, label=label)

# =============================
# Format both plots
# =============================
ax_tension.set_xlabel("Axial Tension (lbf)")
ax_torque.set_xlabel("Torque (lbf·ft)")
ax_tension.set_ylabel("Measured Depth (ft)")

ax_tension.set_title("Tension vs Depth (Survey Mode)")
ax_torque.set_title("Torque vs Depth (Survey Mode)")

for ax in [ax_tension, ax_torque]:
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend(fontsize=8)

plt.tight_layout()

# =============================
# Save downloadable file
# =============================
os.makedirs("results", exist_ok=True)
output_path = "results/tnd_survey_plot_v10.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"✅ Saved plot → {output_path}")
print("📁 You can now download it as a client deliverable.")
