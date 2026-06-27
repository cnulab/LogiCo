# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shutil
import csv
from PIL import Image
import numpy as np

def _mkdirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--data-folder', default='visa_original', type=str,help='the path to downloaded VisA dataset')
parser.add_argument('--save-folder', default='visa', type=str,help='the target path to save the reorganized VisA dataset facilitating data loading in pytorch')
parser.add_argument('--split-file', default='visa_ori/split_csv/1cls.csv', type=str,help='the csv file to split downloaded VisA dataset')

config = parser.parse_args()

split_file = config.split_file
data_folder = config.data_folder
save_folder = config.save_folder

data_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
             'pcb3', 'pcb4', 'pipe_fryum']

for data in data_list:
    train_folder = os.path.join(save_folder, data, 'train')
    test_folder = os.path.join(save_folder, data, 'test')
    mask_folder = os.path.join(save_folder, data, 'ground_truth')

    train_img_good_folder = os.path.join(train_folder, 'ok')
    test_img_good_folder = os.path.join(test_folder, 'ok')
    test_img_bad_folder = os.path.join(test_folder, 'ko')
    test_mask_bad_folder = os.path.join(mask_folder, 'ko')

    _mkdirs_if_not_exists(train_img_good_folder)
    _mkdirs_if_not_exists(test_img_good_folder)
    _mkdirs_if_not_exists(test_img_bad_folder)
    _mkdirs_if_not_exists(test_mask_bad_folder)

with open(split_file, 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        object, set, label, image_path, mask_path = row
        if label == 'normal':
            label = 'ok'
        else:
            label = 'ko'
        image_name = image_path.split('/')[-1]
        mask_name = mask_path.split('/')[-1]
        img_src_path = os.path.join(data_folder, image_path)
        msk_src_path = os.path.join(data_folder, mask_path)
        img_dst_path = os.path.join(save_folder, object, set, label, image_name)
        msk_dst_path = os.path.join(save_folder, object, 'ground_truth', label, mask_name)
        shutil.copyfile(img_src_path, img_dst_path)
        if set == 'test' and label == 'ko':
            mask = Image.open(msk_src_path)

            # binarize mask
            mask_array = np.array(mask)
            mask_array[mask_array != 0] = 255
            mask = Image.fromarray(mask_array)

            mask.save(msk_dst_path)
