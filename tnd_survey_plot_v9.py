import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

print("🔍 Loading Torque & Drag results for plotting...")

# -----------------------------
# Locate results file
# -----------------------------
file_path = "results/results_tnd_v9.csv"

if not os.path.exists(file_path):
    print(f"❌ Results file not found: {file_path}")
    sys.exit(1)

df = pd.read_csv(file_path)
print(f"✅ Loaded {len(df)} rows from {file_path}")

# -----------------------------
# Identify Depth Column
# -----------------------------
depth_col = next(
    (c for c in df.columns if any(k in c.lower() for k in ["depth", "md", "measured_depth"])),
    None
)

if depth_col is None:
    print("❌ No depth or MD column found in the dataset.")
    print("Available columns:", list(df.columns))
    sys.exit(1)

print(f"📏 Using '{depth_col}' as depth column.")

# -----------------------------
# Identify Hookload and Torque Columns
# -----------------------------
hook_cols = [c for c in df.columns if any(x in c.lower() for x in ["pickup", "slackoff", "rotating"])]
torque_cols = [c for c in df.columns if "torque" in c.lower()]

if not hook_cols:
    print("⚠️ No hookload columns found. Expected columns like Pickup_klbf, Slackoff_klbf, Rotating_klbf.")
if not torque_cols:
    print("⚠️ No torque columns found. Expected column like Torque_ftlb.")

# -----------------------------
# Plot Hookload vs Depth
# -----------------------------
if hook_cols:
    plt.figure(figsize=(8, 10))
    for col in hook_cols:
        plt.plot(df[col], df[depth_col], label=col)
    plt.gca().invert_yaxis()
    plt.xlabel("Hookload (klbf)")
    plt.ylabel("Measured Depth (ft)")
    plt.title("Hookload vs Measured Depth (T&D v9)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/hookload_vs_md_v9.png", dpi=300)
    plt.close()
    print("✅ Saved: results/hookload_vs_md_v9.png")

# -----------------------------
# Plot Torque vs Depth
# -----------------------------
if torque_cols:
    plt.figure(figsize=(8, 10))
    for col in torque_cols:
        plt.plot(df[col], df[depth_col], label=col)
    plt.gca().invert_yaxis()
    plt.xlabel("Torque (ft-lbf)")
    plt.ylabel("Measured Depth (ft)")
    plt.title("Torque vs Measured Depth (T&D v9)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/torque_vs_md_v9.png", dpi=300)
    plt.close()
    print("✅ Saved: results/torque_vs_md_v9.png")

print("🎯 Plot generation completed successfully!")
