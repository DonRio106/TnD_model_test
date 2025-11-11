"""
TnD Model v10 (Johancsik-style incremental implementation)
- Uses Johancsik SPE 11380 incremental equations for Pickup/Slackoff/Rotary and Torque
- Per-component weight distribution (BHA from bha_components.csv)
- Torque accumulation per-segment using effective contact radius
- Sweeps friction factors and block weight sensitivity
- Outputs CSV + v9 plots and v10 plot overlaying actual field data (if provided)
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# User parameters (editable)
# -----------------------------
mud_weight_ppg = 9.6                    # ppg
rho_mud = mud_weight_ppg * 8.345        # lb/ft^3
rho_steel = 490.0                       # lb/ft^3 (approx)
BF = 1.0 - (rho_mud / rho_steel)        # buoyancy factor

# torque constant (calibration) - keep to scale torque units
torque_coeff = 1.0  # we will not multiply extra; kept as 1.0 to match physical units

# friction sweep (axial FF)
friction_factors = [0.1, 0.2, 0.3, 0.4, 0.5]

# block weight sensitivity: base (klbf) and variations in klbf
base_block_weight_klbs = 37.0
block_weight_variations = [base_block_weight_klbs - 50.0, base_block_weight_klbs, base_block_weight_klbs + 50.0]

# default torque friction (used to accumulate torque along string)
mu_torque_default = 0.15

# section base mu mapping (used to scale axial FF per section type)
section_base_mu = {
    "Casing": 0.10,   # lower friction when cased
    "OpenHole": 0.50  # higher friction in open hole
}

# -----------------------------
# Input files (expected in repo root)
# -----------------------------
SURVEY_FILE = "survey.csv"                 # must contain columns: MD_ft, Incl_deg (degrees)
BHA_FILE = "bha_components.csv"            # must contain: Component, OD_in, ID_in, Length_ft, Weight_lbft
BOREHOLE_FILE = "borehole.csv"             # optional per-section friction/diameter (Top_MD_ft, Bottom_MD_ft, Hole_ID_in, Type)

# Actual data paths (optional, for overlay)
ACTUAL_PATHS = ["input/ActualTnD.csv", "ActualTnD.csv", "input/ActualTnD.csv".replace("input/","")]

# -----------------------------
# Outputs
# -----------------------------
OUTDIR = "results"
os.makedirs(OUTDIR, exist_ok=True)
OUT_CSV = os.path.join(OUTDIR, "results_tnd_v10_johancsik.csv")

# -----------------------------
# Read inputs & validate
# -----------------------------
if not os.path.exists(SURVEY_FILE):
    raise FileNotFoundError(f"Missing {SURVEY_FILE}")
if not os.path.exists(BHA_FILE):
    raise FileNotFoundError(f"Missing {BHA_FILE}")

survey = pd.read_csv(SURVEY_FILE).copy()
bha = pd.read_csv(BHA_FILE).copy()
borehole = pd.read_csv(BOREHOLE_FILE).copy() if os.path.exists(BOREHOLE_FILE) else None

# Normalize column names
survey.columns = [c.strip() for c in survey.columns]
bha.columns = [c.strip() for c in bha.columns]
if borehole is not None:
    borehole.columns = [c.strip() for c in borehole.columns]

# Accept MD and Incl aliases
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

# validate BHA columns
required_bha = ["Component", "OD_in", "ID_in", "Length_ft", "Weight_lbft"]
for c in required_bha:
    if c not in bha.columns:
        raise KeyError(f"bha_components.csv must include column '{c}'")

# coerce numeric
bha["OD_in"] = pd.to_numeric(bha["OD_in"], errors="coerce")
bha["ID_in"] = pd.to_numeric(bha["ID_in"], errors="coerce")
bha["Length_ft"] = pd.to_numeric(bha["Length_ft"], errors="coerce")
bha["Weight_lbft"] = pd.to_numeric(bha["Weight_lbft"], errors="coerce")

# cumulative lengths and total
bha["CumLen_ft"] = bha["Length_ft"].cumsum()
total_string_length = bha["Length_ft"].sum()

# Helper: component properties at MD (string top at 0 going down)
def get_component_props_at_md(md_value):
    """
    Return (weight_lbft, od_in, id_in, component_name) for the component occupying the given MD position
    If md_value > total_string_length -> returns last component properties
    """
    if md_value <= 0:
        return 0.0, np.nan, np.nan, None
    cum_prev = 0.0
    for idx, row in bha.iterrows():
        comp_top = cum_prev
        comp_bot = row["CumLen_ft"]
        if md_value > comp_top and md_value <= comp_bot:
            return float(row["Weight_lbft"]), float(row["OD_in"]), float(row["ID_in"]), row.get("Component", None)
        cum_prev = comp_bot
    last = bha.iloc[-1]
    return float(last["Weight_lbft"]), float(last["OD_in"]), float(last["ID_in"]), last.get("Component", None)

# Precompute segment midpoints and lengths
dL = np.diff(MD)                # N-1 segments
seg_mid = (MD[:-1] + MD[1:]) / 2.0
seg_inc_deg = (INC[:-1] + INC[1:]) / 2.0
seg_inc_rad = np.radians(seg_inc_deg)
seg_n = len(dL)

# Compute buoyant weight per segment (lb/ft) and OD/ID arrays
seg_wt_lbft = np.zeros(seg_n)
seg_OD_in = np.zeros(seg_n)
seg_ID_in = np.zeros(seg_n)
for i in range(seg_n):
    w_air, od, idc, comp = get_component_props_at_md(seg_mid[i])
    seg_wt_lbft[i] = w_air * BF   # buoyant weight per foot
    seg_OD_in[i] = od if (od is not None and not np.isnan(od)) else np.nan
    seg_ID_in[i] = idc if (idc is not None and not np.isnan(idc)) else np.nan

# helper: determine section type and hole diameter at MD (if borehole defined)
def get_section_info_at_md(md_value):
    if borehole is None:
        return "OpenHole", None
    cols = borehole.columns
    top_col = next((c for c in cols if "Top" in c and "MD" in c), None)
    bot_col = next((c for c in cols if ("Bottom" in c and "MD" in c) or ("Bot" in c and "MD" in c)), None)
    type_col = next((c for c in cols if c.lower() in ["type", "section", "name"]), None)
    holeid_col = next((c for c in cols if "Hole" in c and "ID" in c), None)
    if top_col and bot_col and type_col:
        sel = borehole[(borehole[top_col] <= md_value) & (borehole[bot_col] >= md_value)]
        if not sel.empty:
            hole_id = float(sel[holeid_col].iloc[0]) if holeid_col in sel.columns else None
            return str(sel[type_col].iloc[0]), hole_id
    return "OpenHole", (float(borehole[holeid_col].iloc[-1]) if (borehole is not None and holeid_col in borehole.columns) else None)

# -----------------------------
# Core Johancsik incremental computation
# -----------------------------
rows = []

for bw in block_weight_variations:
    for ff in friction_factors:
        mu_axial_global = ff
        mu_torque = mu_torque_default

        # iterate MD points as "string length in hole" (MD[k])
        for k in range(1, N):
            md_k = MD[k]
            # string length in hole limited by total string length
            string_len_in_hole = min(md_k, total_string_length)

            # include segments whose top < string_len_in_hole
            included = []
            for m in range(seg_n):
                seg_top = MD[m]
                seg_bot = MD[m+1]
                if seg_top >= string_len_in_hole:
                    break
                effective_dL = min(seg_bot, string_len_in_hole) - seg_top
                if effective_dL <= 0:
                    continue
                included.append((m, effective_dL))

            # initialize accumulators for this md_k
            T_pickup = 0.0
            T_slack = 0.0
            T_rot = 0.0
            torque_cum = 0.0

            # perform incremental per included segment (Johancsik)
            for (m, eff_dL) in included:
                Wb = seg_wt_lbft[m]               # lb/ft (buoyant)
                theta = seg_inc_rad[m]            # radians
                dT_weight = Wb * math.cos(theta) * eff_dL    # axial contribution from weight (lbf)
                normal = Wb * math.sin(theta) * eff_dL       # normal force integrated over eff_dL (lbf)
                # section scaling
                section_type, hole_dia = get_section_info_at_md(seg_mid[m])
                base_mu = section_base_mu.get(section_type, section_base_mu["OpenHole"])
                section_scale = base_mu / 0.3 if 0.3 != 0 else 1.0
                mu_axial_effective = mu_axial_global * section_scale

                # Johancsik pickup/slackoff increments
                dT_PU = dT_weight + mu_axial_effective * normal
                dT_SO = dT_weight - mu_axial_effective * normal
                dT_ROT = dT_weight  # rotary axial ignores axial friction

                T_pickup += dT_PU
                T_slack += dT_SO
                T_rot += dT_ROT

                # torque increment: effective radius - use hole diameter if present else pipe OD contact radius
                od = seg_OD_in[m]
                # if hole diameter known, contact radius = (hole_dia - od)/2 (ft) assumed; otherwise use od/2
                if hole_dia is not None and not np.isnan(hole_dia):
                    # effective contact radius in ft: assume pipe contacts borehole, radius = (hole - od)/2 (in) -> ft
                    r_eff_in = max((hole_dia - od) / 2.0, 0.0) if (od is not None and not np.isnan(od)) else (od/2.0 if od is not None else 0.0)
                else:
                    r_eff_in = (od / 2.0) if (od is not None and not np.isnan(od)) else 0.0
                radius_ft = r_eff_in / 12.0

                # torque increment: μ_torque * normal_per_ft * radius * dL
                # normal (lbf over eff_dL) already computed above as integrated normal -> but torque uses normal per ft * dL -> equivalent
                # using: torque_inc = mu_torque * (Wb * sin theta) * radius_ft * eff_dL
                torque_inc = mu_torque * (Wb * math.sin(theta)) * radius_ft * eff_dL * torque_coeff
                torque_cum += torque_inc

            # convert integrated tensions (lbf) to klbf at surface plus block weight offset
            hook_pickup_klbf = (T_pickup / 1000.0) + bw
            hook_slack_klbf = (T_slack / 1000.0) + bw
            hook_rot_klbf = (T_rot / 1000.0) + bw

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

# Save CSV
results.to_csv(OUT_CSV, index=False)
print(f"✅ Saved results to: {OUT_CSV}")

# -----------------------------
# Plotting v10 (model only)
# -----------------------------
plt.figure(figsize=(8, 10))
for ff in friction_factors:
    subset = results[(results["Friction_Factor"] == ff) & (results["Block_Weight_klbf"] == base_block_weight_klbs)]
    plt.plot(subset["Pickup_klbf"], subset["MD_ft"], linestyle="--", label=f"PU FF={ff}")
    plt.plot(subset["Slackoff_klbf"], subset["MD_ft"], linestyle=":", label=f"SO FF={ff}")
    plt.plot(subset["Rotating_klbf"], subset["MD_ft"], linestyle="-", label=f"ROT FF={ff}")
plt.gca().invert_yaxis()
plt.xlabel("Hookload (klbf)")
plt.ylabel("Measured Depth (ft)")
plt.title(f"Hookload vs MD (Base Block Weight {base_block_weight_klbs} klbf) - Johancsik")
plt.legend(fontsize="small", loc="best")
plt.grid(True)
plt.tight_layout()
v9_plot = os.path.join(OUTDIR, "hookload_vs_md_v9_johancsik.png")
plt.savefig(v9_plot, dpi=300)
plt.close()
print(f"✅ Saved model-only plot: {v9_plot}")

# -----------------------------
# Overlay actual data (v10)
# -----------------------------
# Try to find actual file (support multiple path options)
actual_file = None
for p in ACTUAL_PATHS:
    if os.path.exists(p):
        actual_file = p
        break

if actual_file is not None:
    actual = pd.read_csv(actual_file)
    # normalize columns: remove extra spaces and replace with underscores
    actual.columns = [c.strip().replace(" ", "_") for c in actual.columns]
    # try to detect actual pickup/slack/rotating names (be flexible)
    pick_cols = [c for c in actual.columns if c.lower().replace(" ", "").startswith("pickup") or "pickup" in c.lower()]
    slack_cols = [c for c in actual.columns if "slack" in c.lower()]
    rot_cols = [c for c in actual.columns if "rotat" in c.lower()]

    # choose first matches
    pick_col = pick_cols[0] if pick_cols else None
    slack_col = slack_cols[0] if slack_cols else None
    rot_col = rot_cols[0] if rot_cols else None

    # For overlay, we'll interpolate model results onto actual MDs
    # Build model baseline (use FF=0.3 & base block weight case as main model to compare)
    model_ref = results[(np.isclose(results["Friction_Factor"], 0.3)) & (results["Block_Weight_klbf"] == base_block_weight_klbs)]
    if model_ref.empty:
        model_ref = results[(results["Block_Weight_klbf"] == base_block_weight_klbs)]
    # Prepare interpolation functions
    from scipy.interpolate import interp1d

    # ensure increasing MD for interp
    model_ref_sorted = model_ref.sort_values("MD_ft")
    md_model = model_ref_sorted["MD_ft"].to_numpy()
    # use pickup/slack/rot from model for interpolation if available
    interp_pick = interp1d(md_model, model_ref_sorted["Pickup_klbf"].to_numpy(), bounds_error=False, fill_value=np.nan)
    interp_slack = interp1d(md_model, model_ref_sorted["Slackoff_klbf"].to_numpy(), bounds_error=False, fill_value=np.nan)
    interp_rot = interp1d(md_model, model_ref_sorted["Rotating_klbf"].to_numpy(), bounds_error=False, fill_value=np.nan)

    # Now plot model lines + actual points
    plt.figure(figsize=(10, 6))
    # model lines
    for ff in friction_factors:
        subset = results[(results["Friction_Factor"] == ff) & (results["Block_Weight_klbf"] == base_block_weight_klbs)]
        plt.plot(subset["Pickup_klbf"], subset["MD_ft"], linestyle="--", label=f"Model PU FF={ff}")
        plt.plot(subset["Slackoff_klbf"], subset["MD_ft"], linestyle=":", label=f"Model SO FF={ff}")
        plt.plot(subset["Rotating_klbf"], subset["MD_ft"], linestyle="-", label=f"Model ROT FF={ff}")

    # overlay actual (markers)
    if pick_col is not None:
        plt.scatter(actual["MD_ft"], actual[pick_col], color="k", marker="o", label="Actual Pickup", zorder=10)
    if slack_col is not None:
        plt.scatter(actual["MD_ft"], actual[slack_col], color="k", marker="v", label="Actual Slackoff", zorder=10)
    if rot_col is not None:
        plt.scatter(actual["MD_ft"], actual[rot_col], color="k", marker="s", label="Actual Rotating", zorder=10)

    plt.gca().invert_yaxis()
    plt.xlabel("Hookload (klbf)")
    plt.ylabel("Measured Depth (ft)")
    plt.title("Hookload vs MD — Model (lines) vs Actual (markers)")
    plt.legend(fontsize="small", loc="best")
    plt.grid(True)
    plt.tight_layout()

    v10_plot = os.path.join(OUTDIR, "hookload_vs_md_v10_with_actual.png")
    plt.savefig(v10_plot, dpi=300)
    plt.close()
    print(f"✅ Saved overlay plot with actual data: {v10_plot}")

    # Copy actual csv to results for client delivery
    try:
        actual_copy_path = os.path.join(OUTDIR, os.path.basename(actual_file))
        pd.DataFrame(actual).to_csv(actual_copy_path, index=False)
        print(f"✅ Copied actual data to results: {actual_copy_path}")
    except Exception as e:
        print(f"⚠️ Could not copy actual CSV to results folder: {e}")
else:
    print("⚠️ No actual data file found in configured paths; v10 overlay not produced.")

# -----------------------------
# Additional plots: torque & block-sensitivity as before
# -----------------------------
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
plt.savefig(os.path.join(OUTDIR, "torque_vs_md_v9_johancsik.png"), dpi=300)
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
plt.savefig(os.path.join(OUTDIR, "hookload_blockweight_sensitivity_v10_johancsik.png"), dpi=300)
plt.close()

print("✅ Plots generated in 'results/'")
