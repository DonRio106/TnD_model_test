import math

def torque_and_drag(depth, tension, torque):
    """
    Simple TnD model example.
    depth: list of measured depth (m)
    tension: list of tension at each depth (lbf)
    torque: list of torque at each depth (ft-lbf)
    """
    avg_tension = sum(tension) / len(tension)
    avg_torque = sum(torque) / len(torque)
    gradient = (avg_tension / avg_torque) * math.log10(max(depth))
    return gradient

if __name__ == "__main__":
    depth = [1000, 2000, 3000, 4000]
    tension = [20000, 25000, 30000, 35000]
    torque = [500, 700, 900, 1100]
    result = torque_and_drag(depth, tension, torque)
    print(f"Torque & Drag Gradient: {result:.3f}")
