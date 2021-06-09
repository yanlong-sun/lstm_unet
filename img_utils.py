import numpy as np
import imageio


def imsave(image, path):
    label_colours = [(0, 0, 0), (255, 255, 255)]

    images = np.ones(list(image.shape) + [3])
    for j_, j in enumerate(image):
        for k_, k in enumerate(j):
            if k < 2:
                images[j_, k_] = label_colours[int(k)]
    imageio.imwrite(path, images)