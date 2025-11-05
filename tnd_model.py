# tnd_model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Parameters (customize later)
# -----------------------------
depth = np.linspace(0, 10000, 101)          # ft
inclination = np.linspace(0, 60, 101)       # deg
mud_weight = 9.5                            # ppg
pipe_od = 5.0                               # in
pipe_id = 4.276                             # in
pipe_weight = 19.5                          # lb/ft
friction_factor_slackoff = 0.25
friction_factor_pickup = 0.35

# constants
g = 32.174
rho_mud = mud_weight * 0.052 * 144 / g       # lb/ft³ equivalent
area = np.pi / 4 * (pipe_od ** 2 - pipe_id ** 2) / 144  # ft²

# -----------------------------
# Functions
# -----------------------------
def tension_profile(depth, inc, friction):
    """Approximate axial tension considering friction."""
    W = pipe_weight * np.cos(np.radians(inc))
    T = np.zeros_like(depth)
    for i in range(1, len(depth)):
        dL = depth[i] - depth[i - 1]
        T[i] = T[i - 1] + W[i] * dL * (1 + friction)
    return T

def torque_profile(depth, inc, tension):
    """Estimate torque vs depth from tension & geometry."""
    r = pipe_od / 24  # ft radius
    TQ = tension * r * np.sin(np.radians(inc))
    return TQ

# -----------------------------
# Compute Profiles
# -----------------------------
T_slack = tension_profile(depth, inclination, friction_factor_slackoff)
T_pick = tension_profile(depth, inclination, friction_factor_pickup)
TQ = torque_profile(depth, inclination, T_pick)

data = pd.DataFrame({
    "Depth_ft": depth,
    "Incl_deg": inclination,
    "Tension_Slackoff_lbf": T_slack,
    "Tension_Pickup_lbf": T_pick,
    "Torque_lbf_ft": TQ
})

# save CSV
data.to_csv("results_tnd.csv", index=False)

# save plot
plt.figure()
plt.plot(data["Tension_Pickup_lbf"], data["Depth_ft"], label="Pickup")
plt.plot(data["Tension_Slackoff_lbf"], data["Depth_ft"], label="Slackoff")
plt.gca().invert_yaxis()
plt.xlabel("Axial Load (lbf)")
plt.ylabel("Depth (ft)")
plt.legend()
plt.title("Torque & Drag Model")
plt.grid(True)
plt.tight_layout()
plt.savefig("tnd_plot.png", dpi=200)

print("✅ Torque & Drag model run complete.")
