import numpy as np
import string
from skimage import io
import scipy.io as sio
from glob import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt
from exr_utils import load_exr, save_exr


def generate_dmap(head_locs, pmap):
    dmap = np.zeros(pmap.shape[0:2], dtype='float32')
    sigmas = []
    num_heads = head_locs.shape[0]
    for i in range(num_heads):
        x, y = head_locs[i]
        single_density = np.zeros(dmap.shape, dtype='float32')
        single_density[y, x] = 1.
        sigma = 0.2 * pmap[y, x]
        sigmas.append(sigma)
        dmap += gaussian_filter(single_density, sigma)
    dmap_sum = np.sum(dmap)
    norm_factor = num_heads / dmap_sum
    dmap = norm_factor * dmap
    return dmap


def main():
    # fname_image = "100156_A02IndiaWE-03-S20100626080000000E20100626233000000_new.split.5_1.jpg"
    # fname_label = "100156_A02IndiaWE-03-S20100626080000000E20100626233000000_new.split.5_1.mat"
    fname_image = "100156_A02IndiaWE-03-S20100626080000000E20100626233000000_new.split.75_3.jpg"
    fname_label = "100156_A02IndiaWE-03-S20100626080000000E20100626233000000_new.split.75_3.mat"
    fname_pmap = "100156.mat"
    file_pmap = sio.loadmat(fname_pmap)
    file_head_labels = sio.loadmat(fname_label)
    img = io.imread(fname_image)
    pmap = file_pmap["pMap"]
    head_locs = file_head_labels["point_position"]
    dmap = generate_dmap(head_locs, pmap)
    # head_locs = file_head_labels["point_position"]
    # num_heads = head_locs.shape[0]
    # assert num_heads == file_head_labels["point_num"][0,0]
    #
    # labeled_image = img.copy()
    # dmap = np.zeros(img.shape[0:2], dtype='float32')
    # sigmas = []
    # for i in range(num_heads):
    #     x, y = head_locs[i]
    #     labeled_image[y, x+range(-3,4),:] =255
    #     single_density = np.zeros(dmap.shape, dtype='float32')
    #     single_density[y, x] = 1.
    #     sigma = 0.2*pmap[y,x]
    #     sigmas.append(sigma)
    #     dmap += gaussian_filter(single_density, sigma)
    # dmap_sum = np.sum(dmap)
    # norm_factor = num_heads/dmap_sum
    # dmap = norm_factor * dmap

    plt.figure(1)
    plt.imshow(img)
    plt.figure(2)
    plt.imshow(dmap)
    save_exr("{}.exr".format(fname_image),dmap)
    plt.show()
    return

if __name__ == "__main__":
    main()
