# tnd_model_v3.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------
# USER PARAMETERS
# -----------------------------------------------------
mud_weight = 9.5  # ppg
friction_factors = [0.1, 0.2, 0.3, 0.4, 0.5]
block_weight_klbs = 25  # klbs

# Drillstring section definitions
sections = [
    {"name": "Drill Collar", "od": 8.0, "id": 3.25, "weight": 65.7, "start_md": 0, "end_md": 1000},
    {"name": "HWDP",         "od": 5.0, "id": 3.0,  "weight": 25.0, "start_md": 1000, "end_md": 3000},
    {"name": "Drill Pipe",   "od": 5.5, "id": 4.778, "weight": 21.9, "start_md": 3000, "end_md": 10000}
]

# -----------------------------------------------------
# LOAD SURVEY DATA
# -----------------------------------------------------
survey = pd.read_csv("survey.csv")
survey = survey.sort_values("MD_ft").reset_index(drop=True)
depth = survey["MD_ft"].values
inc = survey["Incl_deg"].values
azi = survey["Azi_deg"].values if "Azi_deg" in survey.columns else np.zeros_like(depth)

# -----------------------------------------------------
# BUOYANCY CORRECTION
# -----------------------------------------------------
steel_density_ppg = 65.5
bf = 1 - (mud_weight / steel_density_ppg)

# -----------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------
def get_section(md, sections):
    """Return section weight and OD/ID based on depth."""
    for s in sections:
        if s["start_md"] <= md <= s["end_md"]:
            return s["weight"], s["od"], s["id"], s["name"]
    return sections[-1]["weight"], sections[-1]["od"], sections[-1]["id"], sections[-1]["name"]

def tension_profile(depth, inc, mu, bf, sections):
    """Compute tension considering buoyancy + section properties."""
    T = np.zeros_like(depth)
    sec_list = []
    for i in range(1, len(depth)):
        dL = depth[i] - depth[i - 1]
        wt, od, id_, name = get_section(depth[i], sections)
        eff_wt = wt * bf
        axial_load = eff_wt * np.cos(np.radians(inc[i])) * dL * (1 + mu)
        T[i] = T[i - 1] + axial_load
        sec_list.append(name)
    return T, sec_list

def torque_profile(depth, inc, mu, bf, sections):
    """Compute torque buildup."""
    torque = np.zeros_like(depth)
    for i in range(1, len(depth)):
        dL = depth[i] - depth[i - 1]
        wt, od, id_, name = get_section(depth[i], sections)
        eff_wt = wt * bf
        radius = (od / 12) / 2
        torque[i] = torque[i - 1] + eff_wt * np.sin(np.radians(inc[i])) * dL * mu * radius
    return torque

def hookload(surface_tension, block_klbs):
    return surface_tension + block_klbs * 1000

# -----------------------------------------------------
# MAIN COMPUTATION
# -----------------------------------------------------
results = pd.DataFrame({"MD_ft": depth, "Incl_deg": inc})
plt.figure(figsize=(12, 8))

for mu in friction_factors:
    tension, sections_used = tension_profile(depth, inc, mu, bf, sections)
    torque = torque_profile(depth, inc, mu, bf, sections)
    surface_load = hookload(tension[-1], block_weight_klbs)

    results[f"Tension_mu_{mu}"] = tension
    results[f"Torque_mu_{mu}"] = torque

    print(f"μ={mu:.1f} → Hookload={surface_load/1000:.2f} klbf | Max Torque={max(torque)/1000:.2f} klbf-ft")

    plt.subplot(1, 2, 1)
    plt.plot(tension, depth, label=f"μ={mu}")

    plt.subplot(1, 2, 2)
    plt.plot(torque, depth, label=f"μ={mu}")

# -----------------------------------------------------
# PLOT CONFIGURATION
# -----------------------------------------------------
results.to_csv("results_tnd_survey.csv", index=False)

plt.subplot(1, 2, 1)
plt.gca().invert_yaxis()
plt.xlabel("Axial Tension (lbf)")
plt.ylabel("Measured Depth (ft)")
plt.title("Tension vs Depth (from Survey)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.gca().invert_yaxis()
plt.xlabel("Torque (lbf-ft)")
plt.ylabel("Measured Depth (ft)")
plt.title("Torque vs Depth")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("tnd_survey_plot.png", dpi=200)

print("✅ Full T&D model complete using wellbore trajectory & multi-sections.")
