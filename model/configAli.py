import os
import os.path
import sys
import numpy as np

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

class ConfigAli:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')

    model = 'CPN50'

    lr = 5e-4
    lr_gamma = 0.5
    lr_dec_epoch = list(range(5,40,5))


    batch_size = 12
    weight_decay = 1e-5

    num_class = 24
    # img_path = os.path.join(root_dir, 'data', 'COCO2017', 'train2017')
    symmetry = [(0, 1), (3, 4), (5, 6), (7, 8), (9, 11), (10, 12), (13, 14), (15, 16), (17, 18), (20, 22), (21, 23)]

    # data augmentation setting
    bbox_extend_factor = (0.1, 0.15) # x, y
    scale_factor=(0.7, 1.35)
    rot_factor=30

    pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB
    data_shape = (512, 512) #height, width
    output_shape = (128, 128) #height, width
    gaussain_kernel = (13, 13)
    #
    gk15 = (23, 23)
    gk11 = (17, 17)
    gk9 = (13, 13)
    gk7 = (9, 9)
    # gt_path = os.path.join(root_dir, 'data', 'COCO2017', 'annotations', 'COCO_2017_train.json')

cfg = ConfigAli()
add_pypath(cfg.root_dir)

