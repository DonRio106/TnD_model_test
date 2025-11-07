"""
TnD Model v9
- Incremental MD-step T&D integration (API-style)
- Per-component weight distribution (BHA from bha_components.csv)
- Pickup (pulling up) and Slack-off (letting down) computed separately
- Rotary axial ignores axial friction (no axial FF in rotary), but torque still uses friction
- Torque accumulation computed segment-by-segment (uses component OD for radius)
- Sweeps friction factors and block weight sensitivity
- Outputs CSV + plots into results/
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# User parameters (editable)
# -----------------------------
mud_weight_ppg = 8.5                    # ppg
rho_mud = mud_weight_ppg * 8.345        # lb/ft^3
rho_steel = 490.0                       # lb/ft^3 (approx)
BF = 1.0 - (rho_mud / rho_steel)        # buoyancy factor

# torque constant (calibration)
torque_coeff = 0.0015  # multiplies (mu * normal * radius * dL)

# friction sweep (user wants to explore axial FF 0.1-0.5)
friction_factors = [0.1, 0.2, 0.3, 0.4, 0.5]

# block weight sensitivity: base (klbf) and variations in klbf
base_block_weight_klbs = 25.0
block_weight_variations = [base_block_weight_klbs - 50.0, base_block_weight_klbs, base_block_weight_klbs + 50.0]

# default torque friction (used to accumulate torque along string)
mu_torque_default = 0.15

# section base mu mapping (used to scale axial FF per section type)
# We'll treat the friction factor sweep as a user "global" scale.
# Effective axial mu = ff * (section_base_mu / 0.3)
section_base_mu = {
    "Casing": 0.15,   # lower friction when cased
    "OpenHole": 0.30  # higher friction in open hole
}

# -----------------------------
# Input files (expected in repo root)
# -----------------------------
SURVEY_FILE = "survey.csv"                 # must contain columns: MD_ft, Incl_deg (degrees)
BHA_FILE = "bha_components.csv"            # must contain: Component, OD_in, ID_in, Length_ft, Weight_lbft
BOREHOLE_FILE = "borehole.csv"             # optional per-section friction/diameter (Top_MD_ft, Bottom_MD_ft, Hole_ID_in, Type)

# -----------------------------
# Outputs
# -----------------------------
OUTDIR = "results"
os.makedirs(OUTDIR, exist_ok=True)
OUT_CSV = os.path.join(OUTDIR, "results_tnd_v9.csv")

# -----------------------------
# Helper: read & validate inputs
# -----------------------------
if not os.path.exists(SURVEY_FILE):
    raise FileNotFoundError(f"Missing {SURVEY_FILE}")
if not os.path.exists(BHA_FILE):
    raise FileNotFoundError(f"Missing {BHA_FILE}")

survey = pd.read_csv(SURVEY_FILE).copy()
bha = pd.read_csv(BHA_FILE).copy()
borehole = pd.read_csv(BOREHOLE_FILE).copy() if os.path.exists(BOREHOLE_FILE) else None

# normalize column names (strip spaces)
survey.columns = [c.strip() for c in survey.columns]
bha.columns = [c.strip() for c in bha.columns]
if borehole is not None:
    borehole.columns = [c.strip() for c in borehole.columns]

# detect required survey cols
if "MD_ft" not in survey.columns and "MD" in survey.columns:
    survey = survey.rename(columns={"MD": "MD_ft"})
if "Incl_deg" not in survey.columns and "Inclination" in survey.columns:
    survey = survey.rename(columns={"Inclination": "Incl_deg"})
if "MD_ft" not in survey.columns or "Incl_deg" not in survey.columns:
    raise KeyError("survey.csv must contain 'MD_ft' and 'Incl_deg' columns (or aliases).")

survey = survey.sort_values("MD_ft").reset_index(drop=True)
MD = survey["MD_ft"].to_numpy(dtype=float)
INC = survey["Incl_deg"].to_numpy(dtype=float)
N = len(MD)
if N < 2:
    raise ValueError("survey.csv must contain at least 2 rows (MD points).")

# ensure BHA numeric columns exist
required_bha = ["Component", "OD_in", "ID_in", "Length_ft", "Weight_lbft"]
for c in required_bha:
    if c not in bha.columns:
        raise KeyError(f"bha_components.csv must include column '{c}'")

# coerce numeric
bha["OD_in"] = pd.to_numeric(bha["OD_in"], errors="coerce")
bha["ID_in"] = pd.to_numeric(bha["ID_in"], errors="coerce")
bha["Length_ft"] = pd.to_numeric(bha["Length_ft"], errors="coerce")
bha["Weight_lbft"] = pd.to_numeric(bha["Weight_lbft"], errors="coerce")

# compute cumulative lengths of components (top->bottom order assumed in file)
bha["CumLen_ft"] = bha["Length_ft"].cumsum()
total_string_length = bha["Length_ft"].sum()

# Helper: weight density function (lb/ft) and OD at a given MD (midpoint)
def get_component_props_at_md(md_value):
    """
    Return (weight_lbft, od_in, id_in, component_name) for the component occupying the given MD position
    If md_value > total_string_length -> returns zeros (no component)
    """
    if md_value <= 0:
        # above surface -> no weight
        return 0.0, np.nan, np.nan, None
    cum_prev = 0.0
    for idx, row in bha.iterrows():
        comp_top = cum_prev
        comp_bot = row["CumLen_ft"]
        if md_value > comp_top and md_value <= comp_bot:
            return float(row["Weight_lbft"]), float(row["OD_in"]), float(row["ID_in"]), row.get("Component", None)
        cum_prev = comp_bot
    # if deeper than total string length, return last component properties
    last = bha.iloc[-1]
    return float(last["Weight_lbft"]), float(last["OD_in"]), float(last["ID_in"]), last.get("Component", None)

# Precompute per-segment midpoints, dL, inclination in radians at segment (use endpoint i for simplicity)
dL = np.diff(MD)                # length of segment i (between MD[i] and MD[i+1]) -> len N-1
seg_mid = (MD[:-1] + MD[1:]) / 2.0
seg_inc_deg = (INC[:-1] + INC[1:]) / 2.0
seg_inc_rad = np.radians(seg_inc_deg)
seg_n = len(dL)

# For each segment compute weight density (lb/ft) and OD (in) using midpoint mapping
seg_wt_lbft = np.zeros(seg_n)
seg_OD_in = np.zeros(seg_n)
seg_ID_in = np.zeros(seg_n)
for i in range(seg_n):
    w, od, idc, comp = get_component_props_at_md(seg_mid[i])
    seg_wt_lbft[i] = w * BF      # buoyant weight per ft (lb/ft)
    seg_OD_in[i] = od if od is not None and not np.isnan(od) else np.nan
    seg_ID_in[i] = idc if idc is not None and not np.isnan(idc) else np.nan

# Function to determine section type (cased/open) at a given MD
def get_section_type_at_md(md_value):
    if borehole is None:
        return "OpenHole"
    # expect borehole with Top_MD_ft, Bottom_MD_ft, Type columns (or similar)
    # attempt some alias matching
    cols = borehole.columns
    top_col = next((c for c in cols if "Top" in c and "MD" in c), None)
    bot_col = next((c for c in cols if ("Bottom" in c and "MD" in c) or ("Bot" in c and "MD" in c)), None)
    type_col = next((c for c in cols if c.lower() in ["type", "section", "name"]), None)
    holeid_col = next((c for c in cols if "Hole" in c and "ID" in c), None)
    if top_col and bot_col and type_col:
        sel = borehole[(borehole[top_col] <= md_value) & (borehole[bot_col] >= md_value)]
        if not sel.empty:
            return str(sel[type_col].iloc[0])
    return "OpenHole"

# -----------------------------
# Core computation: for each friction factor and block weight
# We'll compute hookloads & torque for each MD point representing string in hole = MD
# For each MD_k: sum over segments m where segment top >= 0 and segment bottom <= MD_k
# -----------------------------
rows = []

for bw in block_weight_variations:
    for ff in friction_factors:
        # ff is the global axial friction sweep - we'll scale it per section
        # torque friction we keep as mu_torque_default (constant)
        mu_axial_global = ff
        mu_torque = mu_torque_default

        # For each MD index k (representing string length = MD[k]), compute sums over segments 0..k-1
        for k in range(1, N):
            md_k = MD[k]
            # find how many full segments are within md_k: segments indices 0 .. k-1 (since segment i spans MD[i] to MD[i+1])
            # but if MD[k] > total_string_length, we cap at last segment index corresponding to string length
            # compute cumulative string length included = min(md_k, total_string_length)
            string_len_in_hole = min(md_k, total_string_length)
            # determine maximum segment index to include based on seg_mid <= string_len_in_hole
            included = []
            # We'll include segments whose top < string_len_in_hole (i.e., MD[i] < string_len_in_hole)
            for m in range(seg_n):
                seg_top = MD[m]
                seg_bot = MD[m+1]
                if seg_top >= string_len_in_hole:
                    break
                # partial segment handling: effective length inside string = min(seg_bot, string_len_in_hole) - seg_top
                effective_dL = min(seg_bot, string_len_in_hole) - seg_top
                if effective_dL <= 0:
                    continue
                included.append((m, effective_dL))

            # initialize accumulators
            T_pickup = 0.0   # lbf
            T_slack = 0.0    # lbf
            T_rot = 0.0      # lbf (rotary axial ignoring friction)
            torque_cum = 0.0 # ft-lbf (accumulate torque from surface down)
            # compute over included segments
            for (m, eff_dL) in included:
                w = seg_wt_lbft[m]      # lb/ft (buoyant)
                inc = seg_inc_rad[m]
                axial_comp = w * math.cos(inc)   # lbf/ft axial component
                normal_comp = w * math.sin(inc)  # lbf/ft normal component
                # section type
                seg_mid_md = seg_mid[m]
                section_type = get_section_type_at_md(seg_mid_md)
                # scale axial mu by section base mu (so ff 0.3 in openhole ~0.3, in casing ~0.15)
                base_mu = section_base_mu.get(section_type, section_base_mu["OpenHole"])
                section_scale = base_mu / 0.3 if 0.3 != 0 else 1.0
                mu_axial_effective = mu_axial_global * section_scale
                # axial friction for this segment (lbf) over effective length
                friction_lbf = normal_comp * mu_axial_effective * eff_dL
                # incremental axial weight (lbf) over effective length
                dW_axial = axial_comp * eff_dL

                # pickup: friction increases required tension
                T_pickup += dW_axial + friction_lbf
                # slackoff: friction opposes weight (reduces increase)
                T_slack += dW_axial - friction_lbf
                # rotary axial: friction ignored for axial
                T_rot += dW_axial

                # torque: use mu_torque and component radius (use seg_OD_in)
                od = seg_OD_in[m]
                if np.isnan(od):
                    # fallback to first BHA OD
                    od = bha["OD_in"].dropna().iloc[0]
                radius_ft = (od / 12.0) / 2.0
                torque_inc = mu_torque * normal_comp * radius_ft * torque_coeff * eff_dL
                torque_cum += torque_inc

            # after summing segments, compute surface hookload = block weight + integrated tension /1000 (klbf)
            hook_pickup_klbf = (T_pickup / 1000.0) + bw
            hook_slack_klbf  = (T_slack  / 1000.0) + bw
            hook_rot_klbf    = (T_rot    / 1000.0) + bw

            rows.append({
                "MD_ft": md_k,
                "Friction_Factor": ff,
                "Block_Weight_klbf": bw,
                "Pickup_klbf": hook_pickup_klbf,
                "Slackoff_klbf": hook_slack_klbf,
                "Rotating_klbf": hook_rot_klbf,
                "Torque_ftlb": torque_cum
            })

# Build results DataFrame
results = pd.DataFrame(rows)

# Save CSV (overwrite)
results.to_csv(OUT_CSV, index=False)
print(f"✅ Saved results to: {OUT_CSV}")

# -----------------------------
# Plotting (same visuals as before)
# -----------------------------
# Hookloads vs MD for base block weight
plt.figure(figsize=(8, 10))
for ff in friction_factors:
    subset = results[(results["Friction_Factor"] == ff) & (results["Block_Weight_klbf"] == base_block_weight_klbs)]
    plt.plot(subset["Pickup_klbf"], subset["MD_ft"], linestyle="--", label=f"PU FF={ff}")
    plt.plot(subset["Slackoff_klbf"], subset["MD_ft"], linestyle=":", label=f"SO FF={ff}")
    plt.plot(subset["Rotating_klbf"], subset["MD_ft"], linestyle="-", label=f"ROT FF={ff}")
plt.gca().invert_yaxis()
plt.xlabel("Hookload (klbf)")
plt.ylabel("Measured Depth (ft)")
plt.title(f"Hookload vs MD (Base Block Weight {base_block_weight_klbs} klbf)")
plt.legend(fontsize="small", loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "hookload_vs_md_v9.png"), dpi=300)
plt.close()

# Torque vs MD for base block weight
plt.figure(figsize=(8, 10))
for ff in friction_factors:
    subset = results[(results["Friction_Factor"] == ff) & (results["Block_Weight_klbf"] == base_block_weight_klbs)]
    plt.plot(subset["Torque_ftlb"], subset["MD_ft"], label=f"FF={ff}")
plt.gca().invert_yaxis()
plt.xlabel("Torque (ft-lbf)")
plt.ylabel("Measured Depth (ft)")
plt.title("Torque vs Measured Depth (Base Block Weight)")
plt.legend(fontsize="small", loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "torque_vs_md_v9.png"), dpi=300)
plt.close()

# Block weight sensitivity (for FF = 0.3)
plt.figure(figsize=(8, 10))
for bw in block_weight_variations:
    subset = results[(results["Block_Weight_klbf"] == bw) & (np.isclose(results["Friction_Factor"], 0.3))]
    plt.plot(subset["Pickup_klbf"], subset["MD_ft"], label=f"Pickup BW={bw} klbf", linestyle="--")
    plt.plot(subset["Slackoff_klbf"], subset["MD_ft"], label=f"Slackoff BW={bw} klbf", linestyle=":")
    plt.plot(subset["Rotating_klbf"], subset["MD_ft"], label=f"Rotating BW={bw} klbf", linestyle="-")
plt.gca().invert_yaxis()
plt.xlabel("Hookload (klbf)")
plt.ylabel("Measured Depth (ft)")
plt.title("Hookload vs MD (Block Weight Sensitivity, FF=0.3)")
plt.legend(fontsize="small", loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "hookload_blockweight_sensitivity_v9.png"), dpi=300)
plt.close()

print("✅ Plots generated in 'results/'")
