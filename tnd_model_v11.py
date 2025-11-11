"""
TnD Model v11 (Johancsik SPE 11380 Implementation)
- Fully incremental Johancsik-style for Pickup/Slackoff/Rotary and Torque
- Per-component weight, OD/ID, buoyancy correction
- Section-dependent friction (cased vs openhole)
- Overlay with ActualTnD.csv for both Hookload and Torque
- Compatible with GitHub Actions & results/ output
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# -----------------------------
# User parameters
# -----------------------------
mud_weight_ppg = 10
rho_mud = mud_weight_ppg * 8.345
rho_steel = 490.0
BF = 1.0 - (rho_mud / rho_steel)

friction_factors = [0.1, 0.2, 0.3, 0.4, 0.5]
base_block_weight_klbs = 35.0
block_weight_variations = [base_block_weight_klbs - 5, base_block_weight_klbs, base_block_weight_klbs + 5]

mu_torque_default = 0.15
section_base_mu = {"Casing": 0.10, "OpenHole": 0.50}
torque_coeff = 1.0

# -----------------------------
# File paths
# -----------------------------
SURVEY_FILE = "survey.csv"
BHA_FILE = "bha_components.csv"
BOREHOLE_FILE = "borehole.csv"
ACTUAL_PATHS = ["ActualTnD.csv", "input/ActualTnD.csv"]

OUTDIR = "results"
os.makedirs(OUTDIR, exist_ok=True)
OUT_CSV = os.path.join(OUTDIR, "results_tnd_v11.csv")

# -----------------------------
# Input validation
# -----------------------------
def read_csv_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

survey = read_csv_safe(SURVEY_FILE)
bha = read_csv_safe(BHA_FILE)
borehole = pd.read_csv(BOREHOLE_FILE) if os.path.exists(BOREHOLE_FILE) else None

survey.columns = [c.strip() for c in survey.columns]
bha.columns = [c.strip() for c in bha.columns]
if borehole is not None:
    borehole.columns = [c.strip() for c in borehole.columns]

if "MD_ft" not in survey.columns and "MD" in survey.columns:
    survey = survey.rename(columns={"MD": "MD_ft"})
if "Incl_deg" not in survey.columns and "Inclination" in survey.columns:
    survey = survey.rename(columns={"Inclination": "Incl_deg"})

MD = survey["MD_ft"].to_numpy(float)
INC = survey["Incl_deg"].to_numpy(float)
N = len(MD)

# -----------------------------
# Build per-segment geometry
# -----------------------------
bha["CumLen_ft"] = bha["Length_ft"].cumsum()
total_len = bha["Length_ft"].sum()

def get_component(md):
    cum = 0
    for _, r in bha.iterrows():
        if md <= r["CumLen_ft"]:
            return r["Weight_lbft"], r["OD_in"], r["ID_in"]
        cum = r["CumLen_ft"]
    r = bha.iloc[-1]
    return r["Weight_lbft"], r["OD_in"], r["ID_in"]

def get_section(md):
    if borehole is None:
        return "OpenHole", None
    top = [c for c in borehole.columns if "Top" in c and "MD" in c][0]
    bot = [c for c in borehole.columns if "Bot" in c and "MD" in c][0]
    typ = [c for c in borehole.columns if "Type" in c or "Section" in c][0]
    hole = [c for c in borehole.columns if "Hole" in c and "ID" in c][0]
    sel = borehole[(borehole[top] <= md) & (borehole[bot] >= md)]
    if not sel.empty:
        return sel[typ].iloc[0], float(sel[hole].iloc[0])
    return "OpenHole", None

dL = np.diff(MD)
seg_mid = (MD[:-1] + MD[1:]) / 2
seg_inc = np.radians((INC[:-1] + INC[1:]) / 2)
seg_n = len(dL)

seg_wt, seg_od, seg_hole = np.zeros(seg_n), np.zeros(seg_n), np.zeros(seg_n)
for i in range(seg_n):
    w, od, _ = get_component(seg_mid[i])
    seg_wt[i] = w * BF
    seg_od[i] = od

# -----------------------------
# Johancsik incremental calc
# -----------------------------
rows = []
for bw in block_weight_variations:
    for mu_ax in friction_factors:
        for k in range(1, N):
            md_k = MD[k]
            L_in = min(md_k, total_len)
            included = [(i, min(MD[i+1], L_in) - MD[i]) for i in range(seg_n) if MD[i] < L_in]

            Tpu, Tso, Trot, torque = 0, 0, 0, 0
            for i, eff_dL in included:
                Wb = seg_wt[i]
                θ = seg_inc[i]
                section, hole_d = get_section(seg_mid[i])
                mu_eff = mu_ax * (section_base_mu.get(section, 0.3) / 0.3)
                dT_W = Wb * math.cos(θ) * eff_dL
                N_force = Wb * math.sin(θ) * eff_dL

                Tpu += dT_W + mu_eff * N_force
                Tso += dT_W - mu_eff * N_force
                Trot += dT_W

                od = seg_od[i]
                r_eff_in = (hole_d - od) / 2 if hole_d else od / 2
                torque += mu_torque_default * Wb * math.sin(θ) * (r_eff_in / 12) * eff_dL * torque_coeff

            rows.append({
                "MD_ft": md_k,
                "Friction_Factor": mu_ax,
                "Block_Weight_klbf": bw,
                "Pickup_klbf": Tpu/1000 + bw,
                "Slackoff_klbf": Tso/1000 + bw,
                "Rotating_klbf": Trot/1000 + bw,
                "Torque_ftlb": torque
            })

results = pd.DataFrame(rows)
results.to_csv(OUT_CSV, index=False)
print(f"✅ Results saved: {OUT_CSV}")

# -----------------------------
# Plotting — Hookload and Torque (Model Only)
# -----------------------------
def plot_model_only():
    plt.figure(figsize=(8,10))
    for ff in friction_factors:
        s = results[(results["Friction_Factor"] == ff) & (results["Block_Weight_klbf"] == base_block_weight_klbs)]
        plt.plot(s["Pickup_klbf"], s["MD_ft"], '--', label=f"PU FF={ff}")
        plt.plot(s["Slackoff_klbf"], s["MD_ft"], ':', label=f"SO FF={ff}")
        plt.plot(s["Rotating_klbf"], s["MD_ft"], '-', label=f"ROT FF={ff}")
    plt.gca().invert_yaxis()
    plt.xlabel("Hookload (klbf)")
    plt.ylabel("Measured Depth (ft)")
    plt.title("Hookload vs MD — Johancsik Model v11")
    plt.legend(fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "hookload_model_v11.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8,10))
    for ff in friction_factors:
        s = results[(results["Friction_Factor"] == ff) & (results["Block_Weight_klbf"] == base_block_weight_klbs)]
        plt.plot(s["Torque_ftlb"], s["MD_ft"], label=f"FF={ff}")
    plt.gca().invert_yaxis()
    plt.xlabel("Torque (ft-lbf)")
    plt.ylabel("Measured Depth (ft)")
    plt.title("Torque vs MD — Johancsik Model v11")
    plt.legend(fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "torque_model_v11.png"), dpi=300)
    plt.close()

plot_model_only()

# -----------------------------
# Overlay with Actual Data (Hookload + Torque)
# -----------------------------
actual_file = next((p for p in ACTUAL_PATHS if os.path.exists(p)), None)
if actual_file:
    actual = pd.read_csv(actual_file)
    actual.columns = [c.strip().replace(" ", "_").lower() for c in actual.columns]

    pick_col = next((c for c in actual.columns if "pickup" in c), None)
    slack_col = next((c for c in actual.columns if "slack" in c), None)
    rot_col = next((c for c in actual.columns if "rotat" in c), None)
    tq_col = next((c for c in actual.columns if "torque" in c), None)

    # Hookload Overlay
    plt.figure(figsize=(10,6))
    for ff in friction_factors:
        s = results[(results["Friction_Factor"]==ff)&(results["Block_Weight_klbf"]==base_block_weight_klbs)]
        plt.plot(s["Pickup_klbf"], s["MD_ft"], '--', label=f"PU FF={ff}")
        plt.plot(s["Slackoff_klbf"], s["MD_ft"], ':', label=f"SO FF={ff}")
        plt.plot(s["Rotating_klbf"], s["MD_ft"], '-', label=f"ROT FF={ff}")
    if pick_col: plt.scatter(actual["md_ft"], actual[pick_col], c='k', marker='o', label="Actual Pickup")
    if slack_col: plt.scatter(actual["md_ft"], actual[slack_col], c='k', marker='v', label="Actual Slackoff")
    if rot_col: plt.scatter(actual["md_ft"], actual[rot_col], c='k', marker='s', label="Actual Rotating")
    plt.gca().invert_yaxis()
    plt.xlabel("Hookload (klbf)")
    plt.ylabel("Measured Depth (ft)")
    plt.title("Hookload vs MD — Model vs Actual")
    plt.legend(fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "hookload_vs_md_v11_overlay.png"), dpi=300)
    plt.close()

    # Torque Overlay
    if tq_col:
        plt.figure(figsize=(10,6))
        for ff in friction_factors:
            s = results[(results["Friction_Factor"]==ff)&(results["Block_Weight_klbf"]==base_block_weight_klbs)]
            plt.plot(s["Torque_ftlb"], s["MD_ft"], label=f"Model FF={ff}")
        plt.scatter(actual["md_ft"], actual[tq_col], c='r', marker='x', label="Actual Torque")
        plt.gca().invert_yaxis()
        plt.xlabel("Torque (ft-lbf)")
        plt.ylabel("Measured Depth (ft)")
        plt.title("Torque vs MD — Model vs Actual")
        plt.legend(fontsize="small")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "torque_vs_md_v11_overlay.png"), dpi=300)
        plt.close()

    print(f"✅ Overlay plots saved in {OUTDIR}")
else:
    print("⚠️ No ActualTnD.csv found — overlay skipped.")

print("✅ v11 complete.")
