import numpy as np
import matplotlib.pyplot as plt

from skimage import io
import os
from matplotlib_scalebar.scalebar import ScaleBar
from multi_match.colorify import multichannel_to_rgb
import multi_match

fig, ax = plt.subplots(2, 2 , figsize=(8, 8))
for i in range(4):
    file_A = os.path.join('data', f'channel_A_setting_{i+1}.tif')
    file_B = os.path.join('data', f'channel_B_setting_{i+1}.tif')

    image_A = io.imread(file_A)
    image_B = io.imread(file_B)

    # # We now can perform a point detection
    # x = multi_match.point_detection(image_A)
    # y = multi_match.point_detection(image_B)
    # ax.scatter(x[:, 0], x[:, 1], c="green", s=29, edgecolors="white", zorder=3)
    # ax.scatter(y[:, 0], y[:, 1], c="magenta", s=29, edgecolors="white", zorder=3)
    # scalebar = ScaleBar(0.04,
    #         None,
    #         length_fraction=0.10,
    #         box_color="black",
    #         color="white",
    #         location="lower right")
    # scalebar.dx = 25
    # ax.add_artist(scalebar)

    background_image, _, __ = multichannel_to_rgb(images=[image_A, image_B], cmaps=['pure_green', 'pure_magenta'])
    ax[i//2, i%2].imshow(background_image)
    ax[i//2, i%2].set_title(f"Setting {i+1}")
plt.tight_layout()
plt.savefig("simulation_settings.png", dpi=300)
plt.show()


# file_A = os.path.join('data', 'channel_A_setting_1.tif')
# file_B = os.path.join('data', 'channel_B_setting_1.tif')
# 
# image_A = io.imread(file_A)
# image_B = io.imread(file_B)
# 
# 
# 
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# 
# background_image, _, __ = multichannel_to_rgb(images=[image_A, image_B], cmaps=['pure_green', 'pure_magenta'])
# ax.imshow(background_image)
# # ax.axis("off")
# # ax.scatter(x[:, 0], x[:, 1], c="green", s=29, edgecolors="white", zorder=3)
# # ax.scatter(y[:, 0], y[:, 1], c="magenta", s=29, edgecolors="white", zorder=3)
# # scalebar = ScaleBar(0.04,
# #         None,
# #         length_fraction=0.10,
# #         box_color="black",
# #         color="white",
# #         location="lower right")
# # scalebar.dx = 25
# # ax.add_artist(scalebar)
# plt.show()
# 
