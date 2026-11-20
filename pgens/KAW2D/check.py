import matplotlib.pyplot as plt
from matplotlib import patches

fig, ax = plt.subplots(figsize=(5, 5))

img = plt.imread("Shear.png")
ax.imshow(img, extent=(-140, 140, -140, 140), zorder=0)  # (left, right, bottom, top)

for cy in (0, 60, 80, 100):
    rect = patches.Rectangle((-2, cy - 2), 4, 4, color="tab:green", zorder=1)
    ax.add_patch(rect)

ax.set_xlim(-140, 140)
ax.set_ylim(-140, 140)
plt.savefig("VDF_boxes.png")
plt.show()