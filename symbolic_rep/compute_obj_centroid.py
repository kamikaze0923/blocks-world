from symbolic_rep.block import prefix
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize, rescale, downscale_local_mean



img_files = os.listdir(os.path.join(prefix, "image"))
img_files.sort()

all_pixels = []
for f in [os.path.join(prefix, 'image', img_file) for img_file in img_files]:
    img_np = plt.imread(f)
    img_np = resize(img_np, (100,150,4))
    assert np.count_nonzero(img_np[:,:,-1] != 1) == 0
    img_np = img_np[:,:,:3]
    print(img_np.shape)
    pixels = np.resize(img_np, (15000, 3))
    print(pixels.shape)
    all_pixels.append(pixels)

    exit(0)
