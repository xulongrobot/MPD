import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import json
import os

# import caas_jupyter_tools as cjt
import pandas as pd

random.seed(1234)
np.random.seed(1234)


def generate_distributed_scene(num_circles=15, radii_choices=(0.125, 0.15), box=(-1, 1), delta=0.05, max_attempts=5000):
    circles = []
    min_x, max_x = box
    for _ in range(num_circles):
        success = False
        for _ in range(max_attempts):
            r = float(random.choice(radii_choices))
            x = random.uniform(min_x + r, max_x - r)
            y = random.uniform(min_x + r, max_x - r)
            # check distance constraint
            if all(np.hypot(x - c["x"], y - c["y"]) >= (r + c["r"] + delta) for c in circles):
                circles.append({"x": x, "y": y, "r": r})
                success = True
                break
        if not success:
            # failed to place circle, regenerate scene
            return generate_distributed_scene(num_circles, radii_choices, box, delta, max_attempts)
    return circles


# Generate 5 dispersed scenes
scenes = [generate_distributed_scene() for _ in range(1)]

# Save files
# os.makedirs("/mnt/data", exist_ok=True)
json_path = "dispersed_circle_scenes_one.json"
npy_path = "dispersed_circle_scenes_one.npy"
with open(json_path, "w") as f:
    json.dump(scenes, f, indent=2)
np.save(npy_path, np.array(scenes, dtype=object), allow_pickle=True)

# Plot
for i, scene in enumerate(scenes, start=1):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title(f"Scene {i} — {len(scene)} circles (dispersed)")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal", "box")
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1])
    for c in scene:
        circ = Circle((c["x"], c["y"]), c["r"], fill=False)
        ax.add_patch(circ)
    plt.show()

# Display first scene as DataFrame
df0 = pd.DataFrame(scenes[0])
df0.index = [f"circle_{i}" for i in range(1, len(df0) + 1)]
# cjt.display_dataframe_to_user("Scene_1_circles_dispersed", df0)

print("Saved dispersed scenes:")
print(" - JSON:", json_path)
print(" - NPY:", npy_path)

for idx, scene in enumerate(scenes, start=1):
    print(f"\nScene {idx}:")
    for c in scene:
        print(f"  x={c['x']:.4f}, y={c['y']:.4f}, r={c['r']:.3f}")
