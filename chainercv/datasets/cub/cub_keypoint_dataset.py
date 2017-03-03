import collections
import numpy as np
import os.path as osp
from skimage.io import imread

from chainercv.datasets.cub.cub_utils import CUBDatasetBase


class CUBKeypointsDataset(CUBDatasetBase):

    def __init__(self, data_dir='auto', mode='train',
                 crop_bbox=True, bgr=True):
        super(CUBKeypointsDataset, self).__init__(
            data_dir=data_dir, crop_bbox=crop_bbox)

        # set mode
        test_images = np.load(
            osp.join(osp.split(osp.abspath(__file__))[0],
                     'configs/test_images.npy'))
        # the original one has ids starting from 1
        test_images = test_images - 1
        train_images = np.setdiff1d(np.arange(len(self.fns)), test_images)
        if mode == 'train':
            self.selected_ids = train_images
        elif mode == 'test':
            self.selected_ids = test_images
        else:
            raise ValueError('invalid mode')

        # load keypoints
        parts_loc_file = osp.join(self.data_dir, 'parts/part_locs.txt')
        keypoints_dict = collections.OrderedDict()
        for loc in open(parts_loc_file):
            values = loc.split()
            id_ = int(values[0]) - 1
            if id_ not in keypoints_dict:
                keypoints_dict[id_] = []
            keypoints = [float(v) for v in values[2:]]
            keypoints_dict[id_].append(keypoints)
        self.keypoints_dict = keypoints_dict

        self.bgr = bgr

    def __len__(self):
        return len(self.selected_ids)

    def get_example(self, i):
        img, keypoints = self.get_raw_data(i)
        if self.bgr:
            img = img[:, :, ::-1]
        img = img.transpose(2, 0, 1)
        return img, keypoints

    def get_raw_data(self, i):
        # this i is transformed to id for the entire dataset
        original_idx = self.selected_ids[i]
        img = imread(osp.join(
            self.data_dir, 'images', self.fns[original_idx]))  # RGB
        keypoints = self.keypoints_dict[original_idx]
        keypoints = np.array(keypoints, dtype=np.float32)

        if self.crop_bbox:
            bbox = self.bboxes[original_idx]  # (x, y, width, height)
            img = img[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
            keypoints[:, :2] = keypoints[:, :2] - np.array([bbox[0], bbox[1]])

        return img, keypoints


if __name__ == '__main__':
    dataset = CUBKeypointsDataset()

    from chainercv.tasks.pixel_correspondence import vis_verts_pairs
    import matplotlib.pyplot as plt

    for i in range(200, 220):
        src_img, src_keys = dataset.get_raw_data(2 * i)
        dst_img, dst_keys = dataset.get_raw_data(2 * i + 1)
        keys = np.stack([src_keys, dst_keys])
        vis_verts_pairs(src_img, dst_img, keys)
        plt.show()