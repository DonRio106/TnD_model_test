# tnd_model_v4.py
"""
TnD Model v4
- Reads survey.csv and config.yaml
- Computes Pick-Up, Slack-Off, Rotating load cases for a range of friction factors
- Applies buoyancy correction and tool-joint weighting
- Outputs CSV and PNGs into results/
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# ---------------------------
# Utility: load config or defaults
# ---------------------------
DEFAULT_CONFIG = {
    "mud_weight_ppg": 9.5,
    "pipe_od_in": 5.5,
    "pipe_id_in": 4.778,
    "pipe_weight_lbft": 21.9,
    "tool_joint_od_in": 6.75,
    "tool_joint_length_ft": 0.8,
    "joint_length_ft": 31.0,
    "block_weight_klbs": 500,
    "friction_factors": [0.1, 0.2, 0.3, 0.4, 0.5],
    # optional sections (list of dicts)
    "sections": [
        {"name": "Drill Collar", "start_md": 0, "end_md": 1000, "od": 8.0, "id": 3.25, "weight": 65.7},
        {"name": "HWDP",         "start_md": 1000, "end_md": 3000, "od": 5.0, "id": 3.0,  "weight": 25.0},
        {"name": "Drill Pipe",   "start_md": 3000, "end_md": 999999, "od": 5.5, "id": 4.778, "weight": 21.9}
    ]
}

def load_config(path="config.yaml"):
    if os.path.exists(path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        # merge defaults
        final = DEFAULT_CONFIG.copy()
        final.update(cfg)
        # if friction list in yaml as string, handle
        if isinstance(final.get("friction_factors"), (str, float, int)):
            final["friction_factors"] = DEFAULT_CONFIG["friction_factors"]
        return final
    else:
        print("⚠️ config.yaml not found — using defaults.")
        return DEFAULT_CONFIG.copy()

cfg = load_config()

# ---------------------------
# Parameters from config
# ---------------------------
mud_weight = float(cfg["mud_weight_ppg"])
pipe_od = float(cfg["pipe_od_in"])
pipe_id = float(cfg["pipe_id_in"])
pipe_weight = float(cfg["pipe_weight_lbft"])
tool_joint_od = float(cfg["tool_joint_od_in"])
tool_joint_length = float(cfg["tool_joint_length_ft"])
joint_length = float(cfg.get("joint_length_ft", 31.0))
block_weight_klbs = float(cfg["block_weight_klbs"])
friction_factors = list(cfg["friction_factors"])
sections = cfg.get("sections", DEFAULT_CONFIG["sections"])

# ---------------------------
# Read survey.csv (auto-detect MD/Inc/Azi column names)
# ---------------------------
survey_path = "survey.csv"
if not os.path.exists(survey_path):
    raise FileNotFoundError("survey.csv not found in repo root. Please add survey.csv")

survey = pd.read_csv(survey_path)

# normalize column names (strip & replace weird chars)
survey.columns = [c.strip() for c in survey.columns]

# helper to detect column by various aliases
def detect_column(df, aliases):
    for a in aliases:
        if a in df.columns:
            return a
    # try lowercase matching
    cols_lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in cols_lower:
            return cols_lower[a.lower()]
    return None

md_col = detect_column(survey, ["MD_ft", "MD", "Measured Depth (ft)", "MeasuredDepth", "MD (ft)"])
inc_col = detect_column(survey, ["Incl_deg", "Inclination", "Incl", "Inc_deg", "Inclination (deg)"])
azi_col = detect_column(survey, ["Azi_deg", "Azimuth", "Azim_deg", "Azimuth (deg)", "Azim"])

if md_col is None or inc_col is None:
    raise KeyError(f"Could not find required MD / Incl columns in survey.csv. Found columns: {list(survey.columns)}")

# rename to standard names
survey = survey.rename(columns={md_col: "MD_ft", inc_col: "Incl_deg"})
if azi_col:
    survey = survey.rename(columns={azi_col: "Azi_deg"})
else:
    survey["Azi_deg"] = 0.0

survey = survey.sort_values("MD_ft").reset_index(drop=True)

depth = survey["MD_ft"].values.astype(float)
incl = survey["Incl_deg"].values.astype(float)
azi = survey["Azi_deg"].values.astype(float)

# compute segment lengths dL between points
dL = np.diff(depth, prepend=depth[0])  # first segment is zero

# ---------------------------
# Buoyancy and tool-joint handling
# ---------------------------
steel_density_ppg = 65.5
buoyancy_factor = 1.0 - (mud_weight / steel_density_ppg)
eff_pipe_wt = pipe_weight * buoyancy_factor

# compute nominal tool-joint weight per ft (approx)
# assume a tool-joint is heavier than pipe; distribute weight per ft by joint spacing:
tj_weight = (tool_joint_length / joint_length) * pipe_weight * 1.2  # 20% heavier per TJ approx
eff_tj_wt = tj_weight * buoyancy_factor

# for torque radius average (pipe + tj)
radius_pipe_ft = (pipe_od / 12.0) / 2.0
radius_tj_ft = (tool_joint_od / 12.0) / 2.0
radius_mean_ft = (radius_pipe_ft + radius_tj_ft) / 2.0

print(f"Buoyancy factor: {buoyancy_factor:.4f}")
print(f"Effective pipe weight (in mud): {eff_pipe_wt:.3f} lb/ft")
print(f"Effective tool-joint weight (distributed): {eff_tj_wt:.3f} lb/ft")
print("Friction factors:", friction_factors)
print("Using survey rows:", len(depth))

# ---------------------------
# Helper: get section properties at MD
# ---------------------------
def get_section_props(md_val):
    # sections defined by start/end MD in config
    for s in sections:
        if s["start_md"] <= md_val <= s["end_md"]:
            od = float(s.get("od", pipe_od))
            wt = float(s.get("weight", pipe_weight))
            # apply buoyancy to weight per ft for this section
            return {"od": od, "weight": wt * buoyancy_factor, "name": s.get("name", "section")}
    # default last
    s = sections[-1]
    return {"od": float(s.get("od", pipe_od)), "weight": float(s.get("weight", pipe_weight)) * buoyancy_factor, "name": s.get("name", "section")}

# ---------------------------
# Prepare results DataFrame
# ---------------------------
results = pd.DataFrame({"MD_ft": depth, "Incl_deg": incl, "Azi_deg": azi})

# For plotting use distinct colormap ranges
# pick-up: red shades, slack-off: blue shades, rotating: green shades
reds = plt.cm.Reds(np.linspace(0.5, 1.0, len(friction_factors)))
blues = plt.cm.Blues(np.linspace(0.5, 1.0, len(friction_factors)))
greens = plt.cm.Greens(np.linspace(0.4, 0.9, len(friction_factors)))

# We'll build matrices: rows=points, cols=cases
pu_matrix = np.zeros((len(depth), len(friction_factors)))
so_matrix = np.zeros_like(pu_matrix)
rot_matrix = np.zeros_like(pu_matrix)
torque_matrix = np.zeros_like(pu_matrix)

# ---------------------------
# Core computation loop
# ---------------------------
for j, mu in enumerate(friction_factors):
    # initialize
    PU = np.zeros(len(depth))
    SO = np.zeros(len(depth))
    ROT = np.zeros(len(depth))
    TQ = np.zeros(len(depth))

    # Loop depth index from top (surface) to bottom
    for i in range(1, len(depth)):
        seg_md = depth[i]
        seg_inc = incl[i]
        seg_dL = dL[i] if dL[i] > 0 else 0.0

        props = get_section_props(seg_md)
        eff_wt_ft = props["weight"]  # lb/ft after buoyancy for this section
        radius_ft = (props["od"] / 12.0) / 2.0

        # axial deadweight contribution for this small segment
        axial = eff_wt_ft * math.cos(math.radians(seg_inc)) * seg_dL  # lbf

        # Pick-up: friction increases load (acts upward)
        delta_PU = axial * (1.0 + mu)
        PU[i] = PU[i-1] + delta_PU

        # Slack-off: friction reduces resisting load (acts downward) -> smaller increment
        # we simulate by delta = axial*(1 - mu)
        delta_SO = axial * max(0.0, (1.0 - mu))
        SO[i] = SO[i-1] + delta_SO

        # Rotating: take average incremental effect (mid-case)
        delta_ROT = 0.5 * (delta_PU + delta_SO)
        ROT[i] = ROT[i-1] + delta_ROT

        # Torque: incremental torque due to friction * normal ~ eff_wt * sin(inc)
        # simplified: dTorque = mu * eff_wt * sin(inc) * dL * radius
        dTorque = mu * eff_wt_ft * math.sin(math.radians(seg_inc)) * seg_dL * radius_ft
        TQ[i] = TQ[i-1] + dTorque

    # add block weight to surface (convert klbs->lbf)
    block_lbf = block_weight_klbs * 1000.0
    PU_surface = PU[-1] + block_lbf
    SO_surface = SO[-1] + block_lbf
    ROT_surface = ROT[-1] + block_lbf

    print(f"mu={mu:.2f} | PU_surface={PU_surface/1000.0:.2f} klbf | SO_surface={SO_surface/1000.0:.2f} klbf | ROT_surface={ROT_surface/1000.0:.2f} klbf | Max Torque={TQ.max():.0f} lbf-ft")

    # store in matrices & results DF
    pu_matrix[:, j] = PU
    so_matrix[:, j] = SO
    rot_matrix[:, j] = ROT
    torque_matrix[:, j] = TQ

    # add columns to results DataFrame
    results[f"PU_mu_{mu:.2f}"] = PU
    results[f"SO_mu_{mu:.2f}"] = SO
    results[f"ROT_mu_{mu:.2f}"] = ROT
    results[f"TQ_mu_{mu:.2f}"] = TQ

# ---------------------------
# Save CSV and Plots
# ---------------------------
outdir = "results"
os.makedirs(outdir, exist_ok=True)
csv_path = os.path.join(outdir, "results_tnd_loadcases.csv")
plot_path = os.path.join(outdir, "tnd_loadcases_plot.png")
torque_plot_path = os.path.join(outdir, "tnd_torque_plot.png")

results.to_csv(csv_path, index=False)
print(f"Saved results CSV → {csv_path}")

# Plot: pick-up and slack-off and rotating on same figure (left: loads, right: torque)
fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=True)

ax_load = axes[0]
# plot PU (reds)
for j, mu in enumerate(friction_factors):
    ax_load.plot(pu_matrix[:, j] / 1000.0, depth, color=reds[j], linestyle='-', label=f"PU μ={mu:.2f}")
# plot SO (blues)
for j, mu in enumerate(friction_factors):
    ax_load.plot(so_matrix[:, j] / 1000.0, depth, color=blues[j], linestyle='--', label=f"SO μ={mu:.2f}")
# plot ROT (greens)
for j, mu in enumerate(friction_factors):
    ax_load.plot(rot_matrix[:, j] / 1000.0, depth, color=greens[j], linestyle='-.', label=f"ROT μ={mu:.2f}")

ax_load.invert_yaxis()
ax_load.set_xlabel("Load (klbf)")
ax_load.set_ylabel("Measured Depth (ft)")
ax_load.set_title("Pick-up (solid), Slack-off (dashed), Rotating (dash-dot)")
ax_load.grid(True)
# only show one legend with reasonable entries (pick representative mu values)
# build custom handles/labels to avoid SLOTS huge legend:
handles, labels = ax_load.get_legend_handles_labels()
# reduce repeated labels by mapping unique labels
unique = dict(zip(labels, handles))
ax_load.legend(unique.values(), unique.keys(), fontsize='small', loc='best')

# Plot: torque
ax_tq = axes[1]
for j, mu in enumerate(friction_factors):
    ax_tq.plot(torque_matrix[:, j] / 1000.0, depth, label=f"TQ μ={mu:.2f}")
ax_tq.invert_yaxis()
ax_tq.set_xlabel("Torque (k lbf-ft)")
ax_tq.set_title("Torque vs Depth")
ax_tq.grid(True)
ax_tq.legend(fontsize='small', loc='best')

plt.tight_layout()
fig.savefig(plot_path, dpi=200)
print(f"Saved loadcases plot → {plot_path}")

# Also save torque-only figure (bigger)
plt.figure(figsize=(6, 9))
for j, mu in enumerate(friction_factors):
    plt.plot(torque_matrix[:, j] / 1000.0, depth, label=f"TQ μ={mu:.2f}")
plt.gca().invert_yaxis()
plt.xlabel("Torque (k lbf-ft)")
plt.ylabel("Measured Depth (ft)")
plt.title("Torque vs Depth (per friction)")
plt.grid(True)
plt.legend(fontsize='small', loc='best')
plt.tight_layout()
plt.savefig(torque_plot_path, dpi=200)
print(f"Saved torque-only plot → {torque_plot_path}")

print("✅ TnD loadcases completed and saved in results/")

# End of script
