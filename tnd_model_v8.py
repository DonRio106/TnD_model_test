# tnd_model_v8.py
"""
TnD Model v8 (final update)
- Pickup/Slack-off include axial friction (mu_pickup, mu_slackoff)
- Rotary axial DOES NOT include axial friction (rotary clears friction)
- Torque still computed per-segment (uses mu_torque)
- Inputs: survey.csv, bha_components.csv, borehole.csv
- Outputs: results/results_tnd_v8.csv, plots and config summary
"""
import os
import math
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from datetime import datetime

# -----------------------
# Defaults & Config
# -----------------------
# Friction defaults (you confirmed these)
mu_slackoff_default = 0.25
mu_pickup_default = 0.30
mu_rotary_default = 0.15  # used for torque (rotary axial ignores friction)
torque_coeff_default = 0.0015

mud_weight_ppg = 10.0            # default, can be changed in code or config file later
steel_density_ppg = 65.5         # typical steel density (ppg)
BF = 1.0 - (mud_weight_ppg / steel_density_ppg)  # buoyancy factor

block_weight_klbs_default = 400.0

# -----------------------
# Files & folders
# -----------------------
SURVEY_FILE = "survey.csv"
BHA_FILE = "bha_components.csv"
BOREHOLE_FILE = "borehole.csv"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------
# Helpers: column detection
# -----------------------
def detect_col(df, aliases):
    for a in aliases:
        if a in df.columns:
            return a
    cols_lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in cols_lower:
            return cols_lower[a.lower()]
    return None

# -----------------------
# Load inputs
# -----------------------
for f in (SURVEY_FILE, BHA_FILE, BOREHOLE_FILE):
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing required input file: {f}")

survey = pd.read_csv(SURVEY_FILE)
bha = pd.read_csv(BHA_FILE)
borehole = pd.read_csv(BOREHOLE_FILE)

# Normalize column names & detect required columns
md_col = detect_col(survey, ["MD_ft", "MD", "Measured Depth (ft)", "MD"])
inc_col = detect_col(survey, ["Incl_deg", "Inclination", "Incl", "Inc_deg"])
azi_col = detect_col(survey, ["Azi_deg", "Azimuth", "Azim_deg", "Azim"])

if md_col is None or inc_col is None:
    raise KeyError(f"survey.csv must contain MD and Incl columns. Found: {list(survey.columns)}")

survey = survey.rename(columns={md_col: "MD_ft", inc_col: "Incl_deg"})
if azi_col:
    survey = survey.rename(columns={azi_col: "Azi_deg"})
else:
    survey["Azi_deg"] = 0.0

survey = survey.sort_values("MD_ft").reset_index(drop=True)

# Ensure BHA columns exist; allow defaults if missing
bha_cols_expected = {
    "Component": ["Component", "Name"],
    "OD_in": ["OD_in", "OD", "Outer_Diameter_in"],
    "ID_in": ["ID_in", "ID", "Inner_Diameter_in"],
    "Length_ft": ["Length_ft", "Length", "Len_ft"],
    "Weight_lbft": ["Weight_lbft", "Weight", "Weight_lbf_ft"],
    "YieldStrength_psi": ["YieldStrength_psi", "YieldStrength", "YS_psi"],
    "MU_Torque_ftlb": ["MU_Torque_ftlb", "MU_Torque", "Makeup_Torque_ftlb"]
}

# rename detected BHA columns to standard names
rename_map = {}
for std, aliases in bha_cols_expected.items():
    found = detect_col(bha, aliases)
    if found:
        rename_map[found] = std
bha = bha.rename(columns=rename_map)

# If some numeric columns missing, fill with defaults / reasonable values
if "Component" not in bha.columns:
    bha["Component"] = [f"comp_{i}" for i in range(len(bha))]

for col in ["OD_in", "ID_in", "Length_ft", "Weight_lbft"]:
    if col not in bha.columns:
        raise KeyError(f"bha_components.csv must include column for {col} (or alias). Found: {list(bha.columns)}")

# numeric coercion
for c in ["OD_in", "ID_in", "Length_ft", "Weight_lbft", "YieldStrength_psi", "MU_Torque_ftlb"]:
    if c in bha.columns:
        bha[c] = pd.to_numeric(bha[c], errors="coerce")

# fallback defaults for yield and MU torque if missing
default_yield_map = {
    # component name mapping if present
    "Drill Pipe": 95000,
    "DP": 95000,
    "HWDP": 120000,
    "Drill Collar": 135000,
    "DC": 135000
}
if "YieldStrength_psi" not in bha.columns:
    bha["YieldStrength_psi"] = bha["Component"].map(default_yield_map).fillna(95000)
else:
    bha["YieldStrength_psi"] = bha["YieldStrength_psi"].fillna(bha["Component"].map(default_yield_map).fillna(95000))

if "MU_Torque_ftlb" not in bha.columns:
    # simple default: MU torque proportional to OD*ID*some factor (rough)
    bha["MU_Torque_ftlb"] = (bha["OD_in"].fillna(5.5) * 1000).fillna(35000)
else:
    bha["MU_Torque_ftlb"] = bha["MU_Torque_ftlb"].fillna((bha["OD_in"].fillna(5.5) * 1000).fillna(35000))

# Prepare borehole columns detection
bh_top = detect_col(borehole, ["Top_MD_ft", "Top_MD", "Top"])
bh_bot = detect_col(borehole, ["Bottom_MD_ft", "Bottom_MD", "Bottom"])
bh_hole_id = detect_col(borehole, ["Hole_ID_in", "Hole_ID", "Hole_ID"])
bh_type = detect_col(borehole, ["Type", "Section", "Name"])
bh_casing_id = detect_col(borehole, ["Casing_ID_in", "Casing_ID", "Casing_ID_in"])

if not (bh_top and bh_bot and bh_hole_id and bh_type):
    raise KeyError(f"borehole.csv must contain Top_MD_ft, Bottom_MD_ft, Hole_ID_in, Type columns. Found: {list(borehole.columns)}")

# Standardize borehole columns
borehole = borehole.rename(columns={bh_top: "Top_MD_ft", bh_bot: "Bottom_MD_ft", bh_hole_id: "Hole_ID_in", bh_type: "Type"})
if bh_casing_id:
    borehole = borehole.rename(columns={bh_casing_id: "Casing_ID_in"})
else:
    borehole["Casing_ID_in"] = np.nan

# Ensure numeric
borehole["Top_MD_ft"] = pd.to_numeric(borehole["Top_MD_ft"], errors="coerce")
borehole["Bottom_MD_ft"] = pd.to_numeric(borehole["Bottom_MD_ft"], errors="coerce")
borehole["Hole_ID_in"] = pd.to_numeric(borehole["Hole_ID_in"], errors="coerce")

# -----------------------
# Prepare Bha weights (buoyancy)
# -----------------------
# ensure required numeric columns exist
for c in ["Length_ft", "Weight_lbft"]:
    if c not in bha.columns:
        raise KeyError(f"Missing required BHA column: {c}")

bha["Air_Weight_lb"] = bha["Weight_lbft"] * bha["Length_ft"]  # lb per component
bha["Buoyant_Weight_lb"] = bha["Air_Weight_lb"] * BF
bha["Cum_Length_ft"] = bha["Length_ft"].cumsum()

total_buoyant_weight_klbs = bha["Buoyant_Weight_lb"].sum() / 1000.0

# -----------------------
# Model parameters (user-changeable here or later put in config)
# -----------------------
mu_slackoff = mu_slackoff_default
mu_pickup = mu_pickup_default
mu_torque  = mu_rotary_default  # torque friction use
torque_coeff = torque_coeff_default
block_weight_klbs = block_weight_klbs_default

# -----------------------
# Core calculation
# iterate down survey and compute three loadcases
# -----------------------
rows = []
for ff_case in ["slackoff", "pickup", "rotary"]:
    # select friction for axial (rotary axial ignores friction)
    if ff_case == "slackoff":
        mu_axial = mu_slackoff
    elif ff_case == "pickup":
        mu_axial = mu_pickup
    else:  # rotary
        mu_axial = 0.0

    # torque friction always uses mu_torque
    mu_for_torque = mu_torque

    # cumulative torque (accumulates downhole)
    cum_torque = 0.0

    # start with zero surface tension (we compute incremental)
    T_prev = 0.0

    for i, srow in survey.iterrows():
        md = float(srow["MD_ft"])
        inc = math.radians(float(srow["Incl_deg"]))

        # portion of BHA in hole (components with cum length <= md)
        in_hole = bha[bha["Cum_Length_ft"] <= md]
        if in_hole.empty:
            # still append zeros to results to preserve MD points
            rows.append({
                "MD_ft": md,
                "Loadcase": ff_case,
                "Friction_used_axial": mu_axial,
                "Friction_used_torque": mu_for_torque,
                "Pickup_klbf": np.nan,
                "Slackoff_klbf": np.nan,
                "Rotating_klbf": np.nan,
                "Torque_ftlb": np.nan,
                "MU_Torque_Limit": np.nan,
                "Torque_Ratio": np.nan,
                "Section": None
            })
            continue

        # determine borehole section at this MD
        bh_sel = borehole[(borehole["Top_MD_ft"] <= md) & (borehole["Bottom_MD_ft"] >= md)]
        if not bh_sel.empty:
            hole_dia = float(bh_sel["Hole_ID_in"].iloc[0])
            section_type = bh_sel["Type"].iloc[0]
        else:
            hole_dia = float(borehole["Hole_ID_in"].iloc[-1])
            section_type = borehole["Type"].iloc[-1]

        # compute effective (buoyant) in-hole weight (lb)
        W_eff_lb = in_hole["Buoyant_Weight_lb"].sum()
        # vertical (axial) component (lbf)
        W_axial = W_eff_lb * math.cos(inc)
        # normal component (for friction) (lbf)
        # Using simple approximation normal per segment ~ weight * sin(inc)
        W_normal = W_eff_lb * math.sin(inc)

        # drag (axial friction contribution) - aggregated style
        drag = W_normal * mu_axial

        # incremental axial tension (we integrate simply using whole string weight at point)
        # For static cases, friction increases (pickup) or reduces (slackoff) transmitted axial force
        if ff_case == "pickup":
            # Pickup increases required surface tension compared to previous by axial + friction
            T_surface = T_prev + W_axial + drag
        elif ff_case == "slackoff":
            # Slackoff: friction opposes weight, so less tensile increase
            T_surface = T_prev + W_axial - drag
        else:  # rotary (axial friction ignored)
            T_surface = T_prev + W_axial

        # convert to klbf and add block weight offset (surface reference)
        # Note: T_surface currently an integrated cumulative value (lbf). For interpretability we add block weight.
        # Using a consistent approach: Hookload = block_weight + T_surface/1000
        Pickup_klbf = (T_surface + (mu_pickup if ff_case != "pickup" else 0))*0.0  # placeholder not used
        # Instead compute standardized outputs:
        hook_pickup = (T_surface + drag if ff_case == "pickup" else (W_axial + drag)) / 1000.0 + block_weight_klbs
        hook_slack  = (T_surface - drag if ff_case == "slackoff" else (W_axial - drag)) / 1000.0 + block_weight_klbs
        hook_rot    = (T_surface) / 1000.0 + block_weight_klbs

        # Torque: incremental torque due to friction*normal*radius (ft-lbf)
        radius_ft = (hole_dia / 12.0) / 2.0
        # compute incremental torque component for this MD point
        torque_inc = mu_for_torque * W_normal * radius_ft * torque_coeff
        cum_torque += torque_inc

        # M/U torque limit (average of components in hole)
        mu_limit = in_hole["MU_Torque_ftlb"].mean() if "MU_Torque_ftlb" in in_hole.columns else np.nan
        torque_ratio = cum_torque / mu_limit if mu_limit and mu_limit > 0 else np.nan
        if mu_limit and cum_torque > mu_limit:
            warnings.warn(f"Torque exceeds MU torque at MD {md:.0f} ft for loadcase {ff_case} (cum_torque={cum_torque:.1f} ft-lbf, MU={mu_limit:.1f})")

        # Append row (we keep all three hookload definitions but place correct according to loadcase)
        rows.append({
            "MD_ft": md,
            "Loadcase": ff_case,
            "Friction_used_axial": mu_axial,
            "Friction_used_torque": mu_for_torque,
            "Pickup_klbf": (W_axial + drag) / 1000.0 + block_weight_klbs,
            "Slackoff_klbf": (W_axial - drag) / 1000.0 + block_weight_klbs,
            "Rotating_klbf": (W_axial) / 1000.0 + block_weight_klbs,
            "Torque_ftlb": cum_torque,
            "MU_Torque_Limit": mu_limit,
            "Torque_Ratio": torque_ratio,
            "Section": section_type
        })

        # update previous tension for next increment (simple integrator)
        T_prev = T_surface

# Collate results
df_results = pd.DataFrame(rows)

# pivot so we can easily plot per loadcase & per MD for friction sets,
# but currently Loadcase repeated per MD at same friction settings; for clarity we save raw table
timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
results_csv = os.path.join(RESULT_DIR, f"results_tnd_v8_{timestamp}.csv")
df_results.to_csv(results_csv, index=False)

# Save config summary
summary = {
    "Model": "TnD_v8",
    "Timestamp_UTC": timestamp,
    "Mud_weight_ppg": mud_weight_ppg,
    "Buoyancy_factor": round(BF, 6),
    "Block_weight_klbs": block_weight_klbs,
    "mu_slackoff": mu_slackoff,
    "mu_pickup": mu_pickup,
    "mu_torque": mu_torque,
    "torque_coeff": torque_coeff,
    "Total_buoyant_BHA_klbs": round(total_buoyant_weight_klbs, 3),
    "Survey_file": SURVEY_FILE,
    "BHA_file": BHA_FILE,
    "Borehole_file": BOREHOLE_FILE,
    "Results_file": results_csv
}
with open(os.path.join(RESULT_DIR, f"config_summary_v8_{timestamp}.yml"), "w") as f:
    yaml.safe_dump(summary, f, sort_keys=False)

# -----------------------
# Plotting
# -----------------------
# Create a two-panel figure: left = tensions (Pickup/Slack/Rotary) vs MD; right = torque vs MD
fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)

ax_tension = axes[0]
ax_torque = axes[1]

# For plotting, select base friction values to label lines (we used single mu values in this implementation)
# Plot for each loadcase (we will plot the three loadcases aggregated but they were computed per-case)
for lc, style in [("slackoff", ":",), ("pickup", "--"), ("rotary", "-")]:
    df_lc = df_results[df_results["Loadcase"] == lc]
    # group by MD and friction (here friction is the same per run), but we plotted per-case already
    # for clearer view, plot median across friction groups at each MD (or simply plot all rows; here we plot mean)
    agg = df_lc.groupby("MD_ft").agg({
        "Pickup_klbf": "mean",
        "Slackoff_klbf": "mean",
        "Rotating_klbf": "mean",
    }).reset_index()
    ax_tension.plot(agg["Pickup_klbf"], agg["MD_ft"], linestyle="--" if lc=="pickup" else (":" if lc=="slackoff" else "-"), label=f"Pickup mean ({lc})")
    ax_tension.plot(agg["Slackoff_klbf"], agg["MD_ft"], linestyle=":" if lc=="slackoff" else (":"), alpha=0.6)
    ax_tension.plot(agg["Rotating_klbf"], agg["MD_ft"], linestyle="-" if lc=="rotary" else "-", alpha=0.6)

# Tension axis formatting
ax_tension.invert_yaxis()
ax_tension.set_xlabel("Hookload (klbf)")
ax_tension.set_ylabel("Measured Depth (ft)")
ax_tension.set_title("Hookload vs MD (Pickup / Slackoff / Rotary)")
ax_tension.grid(True)
ax_tension.legend(loc="best", fontsize="small")

# Torque panel: plot torque curves and MU torque limit (per-MD average)
# compute mean torque & mean MU limit per MD
torque_agg = df_results.groupby(["MD_ft"]).agg({
    "Torque_ftlb": "mean",
    "MU_Torque_Limit": "mean"
}).reset_index()
ax_torque.plot(torque_agg["Torque_ftlb"], torque_agg["MD_ft"], label="Torque (mean)", color="tab:orange")
ax_torque.plot(torque_agg["MU_Torque_Limit"], torque_agg["MD_ft"], label="M/U Torque Limit (mean)", linestyle="--", color="red")
ax_torque.invert_yaxis()
ax_torque.set_xlabel("Torque (ft-lbf)")
ax_torque.set_title("Torque vs MD")
ax_torque.grid(True)
ax_torque.legend(loc="best", fontsize="small")

plt.tight_layout()
plot_path = os.path.join(RESULT_DIR, f"tension_and_torque_vs_md_v8_{timestamp}.png")
fig.savefig(plot_path, dpi=200)
plt.close(fig)

# Also save torque-only figure larger
plt.figure(figsize=(6, 10))
plt.plot(torque_agg["Torque_ftlb"], torque_agg["MD_ft"], label="Torque (mean)")
plt.plot(torque_agg["MU_Torque_Limit"], torque_agg["MD_ft"], label="M/U Torque Limit (mean)", linestyle="--")
plt.gca().invert_yaxis()
plt.xlabel("Torque (ft-lbf)")
plt.ylabel("MD (ft)")
plt.title("Torque vs MD")
plt.legend()
plt.grid(True)
torque_plot_path = os.path.join(RESULT_DIR, f"torque_vs_md_v8_{timestamp}.png")
plt.tight_layout()
plt.savefig(torque_plot_path, dpi=200)
plt.close()

print("✅ TnD v8 run complete")
print("  Results CSV:", results_csv)
print("  Plot:", plot_path)
print("  Torque plot:", torque_plot_path)
print("  Config summary saved.")

# End of script
