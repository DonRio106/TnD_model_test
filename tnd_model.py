# tnd_model_v2.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------
# PARAMETERS (adjustable / input zone)
# -----------------------------------------------------
depth = np.linspace(0, 10000, 101)           # ft
inclination = np.linspace(0, 60, 101)        # degrees
mud_weight = 9.5                             # ppg
pipe_od = 5.5                                # in
pipe_id = 4.778                              # in (approx for 5.5” Grade S)
pipe_weight = 21.9                           # lb/ft (air weight)
tool_joint_od = 6.75                         # in (typical for 5.5” drill pipe)
tool_joint_length = 0.8                      # ft (~9.6 in)
joint_length = 31.0                          # ft (API standard)
block_weight_klbs = 500                      # klbs (hook load offset)
friction_factors = [0.1, 0.2, 0.3, 0.4, 0.5] # dimensionless

# -----------------------------------------------------
# DERIVED PARAMETERS
# -----------------------------------------------------
steel_density_ppg = 65.5
buoyancy_factor = 1 - (mud_weight / steel_density_ppg)
effective_pipe_weight = pipe_weight * buoyancy_factor  # lb/ft in mud

# Effective tool joint weight (approx)
tool_joint_weight = (tool_joint_length / joint_length) * pipe_weight * 1.2  # 20% heavier per joint
effective_tj_weight = tool_joint_weight * buoyancy_factor

# Average outer radius for torque arm (ft)
radius_pipe = (pipe_od / 12) / 2
radius_tj = (tool_joint_od / 12) / 2
radius_avg = (radius_pipe + radius_tj) / 2

print(f"Buoyancy Factor = {buoyancy_factor:.3f}")
print(f"Effective Pipe Weight = {effective_pipe_weight:.2f} lb/ft")
print(f"Effective Tool Joint Weight = {effective_tj_weight:.2f} lb/ft")
print("-----------------------------------------------------")

# -----------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------
def tension_profile(depth, inc, friction, eff_pipe_weight):
    """Calculate axial tension vs depth considering buoyancy + friction."""
    W = eff_pipe_weight * np.cos(np.radians(inc))
    T = np.zeros_like(depth)
    for i in range(1, len(depth)):
        dL = depth[i] - depth[i - 1]
        T[i] = T[i - 1] + W[i] * dL * (1 + friction)
    return T

def torque_profile(depth, inc, friction, eff_pipe_weight, radius):
    """Estimate torque vs depth considering friction and weight per ft."""
    W = eff_pipe_weight * np.sin(np.radians(inc))
    torque = np.zeros_like(depth)
    for i in range(1, len(depth)):
        dL = depth[i] - depth[i - 1]
        torque[i] = torque[i - 1] + W[i] * dL * friction * radius
    return torque

def hookload(tension_surface, block_weight_klbs):
    """Add block weight (converted to lbf) to total surface load."""
    return tension_surface + (block_weight_klbs * 1000)

# -----------------------------------------------------
# COMPUTATION LOOP
# -----------------------------------------------------
results = pd.DataFrame({"Depth_ft": depth, "Incl_deg": inclination})
plt.figure(figsize=(12, 8))

for mu in friction_factors:
    tension = tension_profile(depth, inclination, mu, effective_pipe_weight)
    torque = torque_profile(depth, inclination, mu, effective_pipe_weight, radius_avg)
    surface_load = hookload(tension[-1], block_weight_klbs)
    
    results[f"Tension_mu_{mu}"] = tension
    results[f"Torque_mu_{mu}"] = torque

    print(f"μ={mu:.1f} → Hookload: {surface_load/1000:.2f} klbf | Max Torque: {max(torque)/1000:.2f} klbf-ft")

    plt.subplot(1, 2, 1)
    plt.plot(tension, depth, label=f"μ={mu}")
    
    plt.subplot(1, 2, 2)
    plt.plot(torque, depth, label=f"μ={mu}")

# -----------------------------------------------------
# SAVE RESULTS
# -----------------------------------------------------
results.to_csv("results_tnd_fullmodel.csv", index=False)

plt.subplot(1, 2, 1)
plt.gca().invert_yaxis()
plt.xlabel("Axial Tension (lbf)")
plt.ylabel("Depth (ft)")
plt.title(f"Tension vs Depth (Buoyancy Corrected, MW={mud_weight} ppg)")
plt.legend(title="μ")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.gca().invert_yaxis()
plt.xlabel("Torque (lbf-ft)")
plt.ylabel("Depth (ft)")
plt.title("Torque vs Depth")
plt.legend(title="μ")
plt.grid(True)

plt.tight_layout()
plt.savefig("tnd_fullmodel_plot.png", dpi=200)

print("✅ Torque & Drag model complete with tool joints, buoyancy, and multi-friction factors.")
