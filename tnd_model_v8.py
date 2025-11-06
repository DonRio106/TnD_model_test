import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import warnings

# -----------------------------
# CONFIGURATION PARAMETERS
# -----------------------------
mud_weight_ppg = 10.0
rho_mud = mud_weight_ppg * 8.345  # lb/ft3
rho_steel = 490.0  # lb/ft3
block_weight_klbs = 25  # klbf
torque_coeff = 0.0015
friction_factors = [0.1, 0.2, 0.3, 0.4, 0.5]

# -----------------------------
# READ INPUT FILES
# -----------------------------
survey = pd.read_csv("survey.csv")
bha = pd.read_csv("bha_components.csv")
borehole = pd.read_csv("borehole.csv")

survey = survey.sort_values("MD_ft").reset_index(drop=True)
bha["Length_ft"] = pd.to_numeric(bha["Length_ft"], errors="coerce")
bha["Weight_lbft"] = pd.to_numeric(bha["Weight_lbft"], errors="coerce")
bha["OD_in"] = pd.to_numeric(bha["OD_in"], errors="coerce")
bha["ID_in"] = pd.to_numeric(bha["ID_in"], errors="coerce")
bha["YieldStrength_psi"] = pd.to_numeric(bha["YieldStrength_psi"], errors="coerce")
bha["MU_Torque_ftlb"] = pd.to_numeric(bha["MU_Torque_ftlb"], errors="coerce")

# -----------------------------
# CALCULATE BUOYANCY FACTOR
# -----------------------------
BF = 1 - (rho_mud / rho_steel)

# -----------------------------
# EFFECTIVE WEIGHTS
# -----------------------------
bha["Air_Weight_lb"] = bha["Weight_lbft"] * bha["Length_ft"]
bha["Buoyant_Weight_lb"] = bha["Air_Weight_lb"] * BF
bha["Cum_Length_ft"] = bha["Length_ft"].cumsum()

total_buoyant_weight_klbs = bha["Buoyant_Weight_lb"].sum() / 1000

# -----------------------------
# HOOKLOAD & TORQUE CALCULATION
# -----------------------------
hookload_data = []

for ff in friction_factors:
    cum_torque = 0.0
    for i, row in survey.iterrows():
        md = row["MD_ft"]
        inc = math.radians(row["Incl_deg"])

        # Determine portion of string in hole
        in_hole = bha[bha["Cum_Length_ft"] <= md]
        if in_hole.empty:
            continue

        # Determine which borehole section applies
        bh_section = borehole[(borehole["Top_MD_ft"] <= md) & (borehole["Bottom_MD_ft"] >= md)]
        if bh_section.empty:
            hole_dia = borehole["Hole_ID_in"].iloc[-1]
            section_type = "OpenHole"
        else:
            hole_dia = float(bh_section["Hole_ID_in"].iloc[0])
            section_type = bh_section["Type"].iloc[0]

        # Weight and frictional force
        W_eff = in_hole["Buoyant_Weight_lb"].sum()
        W_eff_incl = W_eff * math.cos(inc)
        drag = W_eff * ff * math.sin(inc)

        # Hookloads (klbf)
        pickload = (W_eff_incl + drag) / 1000 + block_weight_klbs
        slackoff = (W_eff_incl - drag) / 1000 + block_weight_klbs
        rotating = (W_eff_incl) / 1000 + block_weight_klbs

        # Torque calc (ft-lbf)
        radius_ft = (hole_dia / 12) / 2
        torque_inc = ff * W_eff_incl * radius_ft * torque_coeff
        cum_torque += torque_inc

        # Limit checks
        mu_limit = in_hole["MU_Torque_ftlb"].mean()
        torque_ratio = cum_torque / mu_limit if mu_limit else 0

        if cum_torque > mu_limit:
            warnings.warn(f"⚠️ Torque exceeds M/U torque limit at MD {md:.0f} ft (FF={ff})")

        hookload_data.append({
            "MD_ft": md,
            "Friction_Factor": ff,
            "Pickup_klbf": pickload,
            "Slackoff_klbf": slackoff,
            "Rotating_klbf": rotating,
            "Torque_ftlb": cum_torque,
            "MU_Torque_Limit": mu_limit,
            "Torque_Ratio": torque_ratio,
            "Section": section_type
        })

results = pd.DataFrame(hookload_data)

# -----------------------------
# SAVE RESULTS
# -----------------------------
os.makedirs("results", exist_ok=True)
csv_path = "results/results_tnd_v8.csv"
results.to_csv(csv_path, index=False)

# -----------------------------
# PLOTS
# -----------------------------
plt.figure(figsize=(8, 10))
for ff in friction_factors:
    subset = results[results["Friction_Factor"] == ff]
    plt.plot(subset["Pickup_klbf"], subset["MD_ft"], "--", label=f"PU FF={ff}")
    plt.plot(subset["Slackoff_klbf"], subset["MD_ft"], ":", label=f"SO FF={ff}")
    plt.plot(subset["Rotating_klbf"], subset["MD_ft"], "-", label=f"ROT FF={ff}")

plt.gca().invert_yaxis()
plt.xlabel("Hookload (klbf)")
plt.ylabel("Measured Depth (ft)")
plt.title("Hookload vs MD (T&D Model v8)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/hookload_vs_md_v8.png", dpi=300)
plt.close()

# Torque vs MD
plt.figure(figsize=(8, 10))
for ff in friction_factors:
    subset = results[results["Friction_Factor"] == ff]
    plt.plot(subset["Torque_ftlb"], subset["MD_ft"], label=f"FF={ff}")

plt.plot(results["MU_Torque_Limit"], results["MD_ft"], "r--", label="M/U Torque Limit")
plt.gca().invert_yaxis()
plt.xlabel("Torque (ft-lbf)")
plt.ylabel("Measured Depth (ft)")
plt.title("Torque vs MD (T&D Model v8)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/torque_vs_md_v8.png", dpi=300)
plt.close()

print(f"✅ Simulation complete! Results saved to: {csv_path}")
print("✅ Charts saved in /results folder.")
