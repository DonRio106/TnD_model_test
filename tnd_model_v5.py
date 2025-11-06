import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# =====================================================
# CONFIGURATION PARAMETERS
# =====================================================
mud_weight_ppg = 9.0
rho_mud = mud_weight_ppg * 8.345  # lb/ft3
rho_steel = 490.0  # lb/ft3
block_weight_klbs = 25  # klbf (block weight)
torque_coeff = 0.0015  # scaling factor for torque

# Friction factors to evaluate
friction_factors = [0.1, 0.2, 0.3, 0.4, 0.5]

# =====================================================
# READ INPUT FILES
# =====================================================
try:
    survey = pd.read_csv("survey.csv")
    bha = pd.read_csv("bha_components.csv")
except FileNotFoundError as e:
    raise SystemExit(f"❌ Missing input file: {e.filename}")

# Clean and sort data
survey = survey.sort_values("MD_ft").reset_index(drop=True)
for col in ["Length_ft", "Weight_lbft", "OD_in", "ID_in"]:
    bha[col] = pd.to_numeric(bha[col], errors="coerce")

# =====================================================
# CALCULATE BUOYANCY FACTOR
# =====================================================
BF = 1 - (rho_mud / rho_steel)

# =====================================================
# COMPUTE BHA WEIGHTS
# =====================================================
bha["Air_Weight_lb"] = bha["Weight_lbft"] * bha["Length_ft"]
bha["Buoyant_Weight_lb"] = bha["Air_Weight_lb"] * BF
bha["Cum_Length_ft"] = bha["Length_ft"].cumsum()
total_buoyant_weight_klbs = bha["Buoyant_Weight_lb"].sum() / 1000

# =====================================================
# COMPUTE HOOKLOADS AND TORQUE
# =====================================================
hookload_data = []

for ff in friction_factors:
    for i, row in survey.iterrows():
        md = row["MD_ft"]
        inc = math.radians(row["Incl_deg"])

        # Portion of BHA inside the hole
        in_hole = bha[bha["Cum_Length_ft"] <= md]
        if in_hole.empty:
            continue

        # Effective weight in hole
        W_eff = in_hole["Buoyant_Weight_lb"].sum()
        W_eff_incl = W_eff * math.cos(inc)

        # Frictional drag
        drag = W_eff * ff * math.sin(inc)

        # Hookload calculations (klbf)
        pickload = (W_eff_incl + drag) / 1000 + block_weight_klbs
        slackoff = (W_eff_incl - drag) / 1000 + block_weight_klbs
        rotating = (W_eff_incl) / 1000 + block_weight_klbs

        # Torque (ft-lbf)
        torque = torque_coeff * W_eff_incl * math.sin(inc)

        hookload_data.append({
            "MD_ft": md,
            "Friction_Factor": ff,
            "Pickup_klbf": pickload,
            "Slackoff_klbf": slackoff,
            "Rotating_klbf": rotating,
            "Torque_ftlb": torque
        })

results = pd.DataFrame(hookload_data)

# =====================================================
# SAVE RESULTS
# =====================================================
os.makedirs("results", exist_ok=True)
results_file = os.path.join("results", "results_tnd_v5.csv")
results.to_csv(results_file, index=False)

# =====================================================
# PLOT HOOKLOAD vs MD
# =====================================================
plt.figure(figsize=(8, 10))
for ff in friction_factors:
    subset = results[results["Friction_Factor"] == ff]
    plt.plot(subset["Pickup_klbf"], subset["MD_ft"], label=f"PU FF={ff}", linestyle="--")
    plt.plot(subset["Slackoff_klbf"], subset["MD_ft"], label=f"SO FF={ff}", linestyle=":")
    plt.plot(subset["Rotating_klbf"], subset["MD_ft"], label=f"ROT FF={ff}", linestyle="-")

plt.gca().invert_yaxis()
plt.xlabel("Hookload (klbf)")
plt.ylabel("Measured Depth (ft)")
plt.title("Hookload vs Measured Depth (T&D Model)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/hookload_vs_md.png", dpi=300)
plt.close()

# =====================================================
# PLOT TORQUE vs MD
# =====================================================
plt.figure(figsize=(8, 10))
for ff in friction_factors:
    subset = results[results["Friction_Factor"] == ff]
    plt.plot(subset["Torque_ftlb"], subset["MD_ft"], label=f"FF={ff}")

plt.gca().invert_yaxis()
plt.xlabel("Torque (ft-lbf)")
plt.ylabel("Measured Depth (ft)")
plt.title("Torque vs Measured Depth (T&D Model)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/torque_vs_md.png", dpi=300)
plt.close()

# =====================================================
# DONE
# =====================================================
print(f"✅ Simulation complete. Results saved to: {results_file}")
print("✅ Hookload and Torque plots generated in /results folder.")
