# ###################################
# Group ID : 203
# Members : Malthe Boelskift, Louis Ildal, Guillermo Gutierrez Bea, Nikolaos Gkloumpos.
# Date : 18/10/2023
# Lecture: 11
# Dependencies: numpy, matplotlib, sklearn
# Python version: 3.11.4
# Functionality: 
# ###################################

import numpy as np
from skimage import data, color
import matplotlib.pyplot as plt
import numba

image_rgb = data.stereo_motorcycle()[0]
image = np.round(np.mean(image_rgb, axis=2, dtype=np.int16))

plt.figure(figsize=(10, 10))
plt.imshow(image, cmap='gray')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.tight_layout()
plt.show()

var = 100

noise = np.asarray(np.round(np.random.normal(loc=0, scale=np.sqrt(var), size=image.shape)), dtype=np.int16)

image_noisy = (image + noise)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image_noisy[i, j] > 255:
            image_noisy[i, j] = 255
        elif image_noisy[i, j] < 0:
            image_noisy[i, j] = 0

plt.figure(figsize=(10, 10))
plt.imshow(image_noisy, cmap='gray')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.tight_layout()
plt.show()

@numba.jit(nopython=True)
def func(_im_noise, _var, _iterations, _w_diff, _max_diff):
    image_buffer = np.zeros((2, _im_noise.shape[0], _im_noise.shape[1]), dtype=_im_noise.dtype)
    image_buffer[0] = _im_noise
    image_buffer[1] = _im_noise

    V_max = _im_noise.shape[0] * _im_noise.shape[1] * (256**2) / (2 * _var) + 4 * _w_diff * _max_diff
    
    for n in range(_iterations):
        s = n % 2
        d = (n + 1) % 2

        for i in range(_im_noise.shape[0]):
            for j in range(_im_noise.shape[1]):
                V_local = V_max
                min_val = -1

                for k in range(256):
                    V_data = (k - _im_noise[i, j])**2 / (2 * _var)
                    V_diff = 0

                    if i > 0:
                        V_diff += min(((k - image_buffer[s, i - 1, j])**2, _max_diff))
                    if i < _im_noise.shape[0] - 1:
                        V_diff += min(((k - image_buffer[s, i + 1, j])**2, _max_diff))
                    if j > 0:
                        V_diff += min(((k - image_buffer[s, i, j - 1])**2, _max_diff))
                    if j < _im_noise.shape[1] - 1:
                        V_diff += min(((k - image_buffer[s, i, j + 1])**2, _max_diff))
                    
                    V_current = V_data + _w_diff * V_diff

                    if V_current < V_local:
                        min_val = k
                        V_local = V_current

                image_buffer[d, i, j] = min_val

    return image_buffer[d]

# Hyperparameters
iterations = 100
w_diff = 0.015
max_diff = 100

image_recon = func(image_noisy, var, iterations, w_diff, max_diff)

plt.figure(figsize=(30, 10))

plt.subplot(131)
plt.title('Original')
plt.imshow(image, cmap='gray')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.tight_layout()

plt.subplot(132)
plt.title('Noisy')
plt.imshow(image_noisy, cmap='gray')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.tight_layout()

plt.subplot(133)
plt.title('Reconstruction')
plt.imshow(image_recon, cmap='gray')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.tight_layout()

plt.show()