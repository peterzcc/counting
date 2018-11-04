import numpy as np
import string
from skimage import io
import scipy.io as sio
from glob import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt
from exr_utils import load_exr, save_exr
import os
import argparse
import hdf5storage

def generate_dmap(head_locs, pmap):
    dmap = np.zeros(pmap.shape[0:2], dtype='float32')
    sigmas = []
    num_heads = head_locs.shape[0]
    for i in range(num_heads):
        x, y = head_locs[i]
        x -= 1
        y -= 1 # change from matlab index to python index
        single_density = np.zeros(dmap.shape, dtype='float32')
        single_density[y, x] = 1.
        sigma = 0.2 * pmap[y, x]
        sigmas.append(sigma)
        dmap += gaussian_filter(single_density, sigma)
    dmap_sum = np.sum(dmap)
    assert not np.isnan(dmap).any()
    if dmap_sum > 0:
        norm_factor = num_heads / dmap_sum
        dmap = norm_factor * dmap
    assert not np.isnan(dmap).any()
    return dmap


def main():
    parser = argparse.ArgumentParser(description='generate dmap.')
    parser.add_argument('--pre', type=str, default='train', help='train, test, etc')
    args = parser.parse_args()
    data_dir = "."
    prefix = args.pre
    label_top_dir = "{}/{}_label".format(data_dir, prefix)
    persp_dir = "{}/{}_perspective".format(data_dir, prefix)
    output_dir = "{}/{}_dmap".format(data_dir, prefix)
    if not os.path.exists(output_dir):
        print("creating dir: {}".format(output_dir))
        os.mkdir(output_dir)
    for label_subd in os.listdir(os.fsencode(label_top_dir)):
        name_label_subd = os.fsdecode(label_subd)
        persp_fname = "{}/{}.mat".format(persp_dir, name_label_subd)
        assert os.path.exists(persp_fname)
        file_pmap = sio.loadmat(persp_fname)
        pmap = file_pmap["pMap"]
        # print(os.fsdecode(label_subd))
        this_scene_path = "{}/{}".format(label_top_dir,name_label_subd)
        for label_name in os.listdir(os.fsencode(this_scene_path)):
            name_label = os.fsdecode(label_name)
            if name_label == "roi.mat":
                continue
            if not name_label.endswith(".mat"):
                continue
            output_path = "{}/{}.exr".format(output_dir, name_label[:-4])
            if os.path.exists(output_path):
                # print("exists: {}".format(output_path))
                continue

            path_label = "{}/{}".format(this_scene_path,name_label)
            file_head_labels = hdf5storage.loadmat(path_label)
            head_locs = np.array(file_head_labels["point_position"]).astype(int)
            # print(head_locs)
            dmap = generate_dmap(head_locs, pmap)

            print(output_path)
            save_exr(output_path, dmap)


def test():
    fname_image = "100156_A02IndiaWE-03-S20100626080000000E20100626233000000_new.split.75_3.jpg"
    fname_label = "200707_C09-04-S20100717083000000E20100717233000000_1_clip1_1.mat"
    fname_pmap = "200707.mat"

    # file_pmap = sio.loadmat(fname_pmap)
    # img = io.imread(fname_image)
    # pmap = file_pmap["pMap"]
    # file_head_labels = hdf5storage.loadmat(fname_label)
    # head_locs = np.array(file_head_labels["point_position"]).astype(int)
    # dmap = generate_dmap(head_locs, pmap)


    # plt.figure(1)
    # plt.imshow(img)
    # plt.figure(2)
    # plt.imshow(dmap)
    # plt.show()
    #
    # save_exr("{}.exr".format(fname_image),dmap)
    roi_fname = "roi.mat"
    roi = hdf5storage.loadmat(roi_fname)
    from PIL import Image, ImageDraw
    w,h = 720,576
    vx = roi["maskVerticesXCoordinates"] - 1
    vy = roi["maskVerticesYCoordinates"] - 1
    img = Image.new('L', (w, h), 0)

    polygon = np.squeeze(np.array([(x,y) for x,y in zip(vx,vy)]))
    polygon = np.round(polygon)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img)*255
    plt.imshow(mask,cmap='gray')
    plt.show()


    return


if __name__ == "__main__":
    main()
    # test()