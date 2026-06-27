import os
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import argparse


def image_info(image_name, category):
    image_name = image_name.replace('__','_')[len(category)+1:]
    subject_id, label, sub_cate,view_id,id =image_name.split('_')
    return subject_id, label,sub_cate,view_id,id

def read_json(json_path):
    with open(json_path,'r+') as f:
        samples =  json.load(f)

    train_samples = {sample['image_path'][sample['image_path'].rfind('/')+1:]:None for sample in samples['train']}
    test_samples = {sample['image_path'][sample['image_path'].rfind('/')+1:]:None for sample in samples['test']}

    return train_samples, test_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preparation')
    parser.add_argument('--data-folder', default='realiad_original', type=str,help='the path to downloaded Real-IAD dataset')
    parser.add_argument('--save-folder', default='realiad', type=str,help='the target path to save the reorganized Real-IAD dataset facilitating data loading in pytorch')
    parser.add_argument('--split-file-root', default='realiad_jsons_sv', type=str)
    config = parser.parse_args()
    
    src_root = config.data_folder
    dst_root = config.save_folder
    split_file_root = config.split_file_root
    
    categories = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'u_block', 'usb',
                 'usb_adaptor',  'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    
    view = 'C1'
    
    for category in categories:
        print(f'start extract {category}')
        src_category_root = os.path.join(src_root, category)
        dst_category_root = os.path.join(dst_root, category)
        os.makedirs(dst_category_root, exist_ok=True)
    
        json_path = os.path.join(split_file_root, f'{category}.json')
        train_samples, test_samples = read_json(json_path)
    
        src_test_root = os.path.join(src_category_root, 'NG')
        
        dst_test_root = os.path.join(dst_category_root,'test')
        dst_ground_truth_root = os.path.join(dst_category_root,'ground_truth')
        dst_train_root = os.path.join(dst_category_root,'train')
        
        for sub_cate in os.listdir(src_test_root):
    
            sub_cate_root = os.path.join(src_test_root, sub_cate)
    
            dst_sub_cate_root = os.path.join(dst_test_root, sub_cate)
            os.makedirs(dst_sub_cate_root, exist_ok=True)
            
            dst_ground_sub_cate_root = os.path.join(dst_ground_truth_root, sub_cate)
            os.makedirs(dst_ground_sub_cate_root, exist_ok=True)
    
            for subject in tqdm(os.listdir(sub_cate_root)):
                subject_root = os.path.join(sub_cate_root, subject)
                view_images = [image  for image in os.listdir(subject_root) if image.endswith('jpg')]
                for view_image in view_images:
                    subject_id, label, sub_cate, view_id, id = image_info(view_image, category)
                    if view_id == view:
                        view_path = os.path.join(subject_root, view_image)
                        mask_path = os.path.join(subject_root, view_image.replace('jpg','png'))
                        if os.path.exists(mask_path):
                            dst_view_path = os.path.join(dst_sub_cate_root, view_image)
                            dst_gt_path = os.path.join(dst_ground_sub_cate_root, view_image.replace('jpg','png'))
                            shutil.copy(view_path,dst_view_path)
                            shutil.copy(mask_path,dst_gt_path)
                        else:
                            if view_image in train_samples:
                                train_ok_root = os.path.join(dst_train_root,'OK')
                                os.makedirs(train_ok_root,exist_ok=True)
                                shutil.copy(view_path,os.path.join(train_ok_root,view_image))
                            else:
                                assert view_image in test_samples
                                test_ok_root = os.path.join(dst_test_root,'OK')
                                os.makedirs(test_ok_root,exist_ok=True)
                                shutil.copy(view_path,os.path.join(test_ok_root,view_image))
    
        src_train_root = os.path.join(src_category_root,'OK')
    
        for subject in tqdm(os.listdir(src_train_root)):
            subject_root = os.path.join(src_train_root, subject)
            view_images = [ image  for image in os.listdir(subject_root) if image.endswith('jpg')]
    
            for view_image in view_images:
    
                if view_image.find(view)!=-1:
                    view_path = os.path.join(subject_root, view_image)
    
                    if view_image in train_samples:
                        train_ok_root = os.path.join(dst_train_root,'OK')
                        os.makedirs(train_ok_root,exist_ok=True)
                        shutil.copy(view_path,os.path.join(train_ok_root,view_image))
                    else:
                        assert view_image in test_samples
                        test_ok_root = os.path.join(dst_test_root,'OK')
                        os.makedirs(test_ok_root,exist_ok=True)
                        shutil.copy(view_path,os.path.join(test_ok_root,view_image))