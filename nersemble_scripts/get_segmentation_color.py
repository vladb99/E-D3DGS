import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def get_rgb(event):
    # Check if click was on the axes
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        rgb = img[y, x, :]
        print(f"RGB values at ({x}, {y}): {rgb}")

# Load the image
img = mpimg.imread('/home/vbratulescu/Downloads/407-tongue-annotations/407/sequences/EXP-6-tongue-1/timesteps/frame_00270/facer_segmentation_masks/color_segmentation_cam_222200040.png')

# Display the image
fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title('Click on the image to get RGB values of a pixel')
fig.canvas.mpl_connect('button_press_event', get_rgb)

plt.show()