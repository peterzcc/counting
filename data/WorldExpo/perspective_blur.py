import numpy as np
import string
from skimage import io
import scipy.io as sio
from glob import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt

# image_list = list(glob('./*.png'))
# sigmadots = 8
#
# pmap = sio.loadmat('vidf1_33_dmap3.mat')
# pmap = pmap['dmap'][0][0]
# pmap = pmap['pmapxy']
#
# for fname in image_list:
#     im = io.imread(fname)
#     count = np.count_nonzero(im[:,:,0])
#     dot = np.zeros(im[:,:,0].shape, dtype='float32')
#     nonzero = np.nonzero(im)
#     nonzero = zip(nonzero[0], nonzero[1])
#     for (x,y) in nonzero:
#         curr_dot = np.zeros(dot.shape, dtype='float32')
#         curr_dot[x,y] = 1.0
#         dot += gaussian_filter(curr_dot, sigmadots/(np.sqrt(pmap[x,y])))
#     countdot = np.sum(dot)
#     print('{0}: {1} (8-bit) {2} (summed)'.format(fname,count,countdot))
#     fname_out = fname.replace('.png','.npy')
#     np.save(fname_out, dot)

def main():
    # fname_image = "100156_A02IndiaWE-03-S20100626080000000E20100626233000000_new.split.5_1.jpg"
    # fname_label = "100156_A02IndiaWE-03-S20100626080000000E20100626233000000_new.split.5_1.mat"
    fname_image = "100156_A02IndiaWE-03-S20100626080000000E20100626233000000_new.split.75_3.jpg"
    fname_label = "100156_A02IndiaWE-03-S20100626080000000E20100626233000000_new.split.75_3.mat"
    fname_pmap = "100156.mat"
    file_pmap = sio.loadmat(fname_pmap)
    pmap = file_pmap["pMap"]
    file_head_labels = sio.loadmat(fname_label)
    head_locs = file_head_labels["point_position"]
    num_heads = head_locs.shape[0]
    assert num_heads == file_head_labels["point_num"][0,0]
    img = io.imread(fname_image)
    labeled_image = img.copy()
    dmap = np.zeros(img.shape[0:2], dtype='float32')
    sigmas = []
    for i in range(num_heads):
        x, y = head_locs[i]
        labeled_image[y, x+range(-3,4),:] =255
        single_density = np.zeros(dmap.shape, dtype='float32')
        single_density[y, x] = 1.
        sigma = 0.2*pmap[y,x]
        sigmas.append(sigma)
        dmap += gaussian_filter(single_density, sigma)
    dmap_sum = np.sum(dmap)
    norm_factor = num_heads/dmap_sum
    dmap = norm_factor * dmap
    plt.figure(1)
    plt.imshow(labeled_image)
    plt.figure(2)
    plt.imshow(dmap)
    # plt.show()
    return

if __name__ == "__main__":
    main()
