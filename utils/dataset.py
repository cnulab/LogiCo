import random
from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
import torch.multiprocessing
import json
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import multi_scale_feat_fusion
import math
from typing import Sequence, List, Tuple
import re


torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train, std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


def gen_crops(h: int, w: int, padding = True) -> List[Tuple[int, int, int, int]]:

    def extract_starts(image_size, patch_size, stride, padding = True):
        if image_size <= patch_size:
            starts = [0, ]
        else:
            starts = list(range(0, image_size, stride))
            for i in range(len(starts)):
                if starts[i] + patch_size > image_size:
                    starts[i] = image_size - patch_size
            starts = sorted(set(starts), key=starts.index)
            if not padding:
                starts = starts[:-1]
        return starts

    AREA_FRACS = [0.05, 0.2, 1.0]
    STRIDE_RATIO = 0.5
    crops = []

    for af in AREA_FRACS:
        side = max(16, int(round(math.sqrt(af) * min(h, w))))
        stride = max(8, int(round(side * STRIDE_RATIO)))
        y_starts_list = extract_starts(h, side, stride, padding)
        x_starts_list = extract_starts(w, side, stride, padding)
        for x in x_starts_list:
            for y in y_starts_list:
                crops.append([x, y, x+side, y+side])
    return crops



@torch.no_grad()
def aggregate_features(encoder, target_layers,
                       feat_stride, feat_dim,
                       data_transform, pil_img, device, padding = True):

    H, W = pil_img.height, pil_img.width

    fmap, hits = None, torch.zeros((H, W), device = device)

    crop_images = []
    points = []

    for (x0, y0, x1, y1) in gen_crops(H, W, padding):
        crop = pil_img.crop((x0, y0, x1, y1))
        crop_t = data_transform(crop).unsqueeze(0).to(device)
        crop_images.append(crop_t)
        points.append([x0, y0, x1, y1])

    crop_images = torch.cat(crop_images)

    en_list = encoder.get_intermediate_layers(crop_images.cuda(), n=target_layers, norm=False)
    toks = multi_scale_feat_fusion(en_list, feat_dim)

    for point, tok in zip(points, toks):

        if fmap is None:
            fmap = torch.zeros((tok.size(1), H, W), device=device)

        x0, y0, x1, y1 = point

        P = tok.size(0)

        w_p = int(round(math.sqrt(P)))
        h_p = P // w_p
        post_grid = tok.movedim(1, 0).reshape(fmap.size(0), h_p, w_p)

        post_grid = F.interpolate(post_grid.unsqueeze(0), size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)[0]
        fmap[:, y0: y1, x0 : x1] += post_grid
        hits[y0: y1, x0: x1] += 1

    feat = (fmap / hits.clamp(min=1))
    feat = F.interpolate(feat.unsqueeze(0), size=(H//feat_stride, W//feat_stride), mode = 'bilinear')[0]
    return feat



def load_anomaly_detection_dataset(dataset, dataset_root, seg_root, split, val_proportion = None):
    assert split in ['train', 'test']

    image_root = os.path.join(dataset_root, split)
    seg_root = os.path.join(seg_root, split)
    gt_root = os.path.join(dataset_root, 'ground_truth')

    samples = []
    defect_types = os.listdir(image_root)
    for defect_type in defect_types:
        if defect_type in ['good', 'ok', 'OK']:
            img_paths = glob.glob(os.path.join(image_root, defect_type) + "/*.png") + \
                        glob.glob(os.path.join(image_root, defect_type) + "/*.jpg") + \
                        glob.glob(os.path.join(image_root, defect_type) + "/*.JPG") + \
                        glob.glob(os.path.join(image_root, defect_type) + "/*.bmp")

            for img_path in img_paths:
                img_name = os.path.basename(img_path)
                seg_path = os.path.join(seg_root, defect_type, re.sub('jpg', 'png', img_name, flags=re.IGNORECASE))
                samples.append([img_path, seg_path, '', defect_type, 0])

        else:
            img_paths = glob.glob(os.path.join(image_root, defect_type) + "/*.png") + \
                        glob.glob(os.path.join(image_root, defect_type) + "/*.jpg") + \
                        glob.glob(os.path.join(image_root, defect_type) + "/*.JPG") + \
                        glob.glob(os.path.join(image_root, defect_type) + "/*.bmp")

            for img_path in img_paths:
                img_name = os.path.basename(img_path)
                seg_path = os.path.join(seg_root, defect_type, re.sub('jpg', 'png', img_name, flags=re.IGNORECASE))

                if dataset in ['visa', 'realiad']:
                    gt_path = os.path.join(gt_root, defect_type, re.sub('jpg', 'png', img_name, flags=re.IGNORECASE))
                elif dataset in ['mvtec']:
                    gt_path = os.path.join(gt_root, defect_type, img_name.replace('.', '_mask.'))
                else:
                    raise RuntimeError

                samples.append([img_path, seg_path, gt_path, defect_type, 1])

    if split == 'train':
        assert val_proportion is not None and val_proportion > 0 and val_proportion < 1
        train_number = int(len(samples) * (1 - val_proportion))
        return samples[:train_number], samples[train_number:]
    else:
        return samples


class LOCODatasetWithSeg(torch.utils.data.Dataset):

    def __init__(self, root, image_size, transform, gt_transform, phase):
        assert phase in ['train', 'validation', 'test']

        self.root = root
        if phase == 'train' or phase == 'validation':
            self.img_path = os.path.join(root, phase)
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')

        self.image_size = image_size
        self.phase = phase
        self.transform = transform
        self.gt_transform = gt_transform

        self.dataset_root, self.category = os.path.split(self.root)
        self.seg_root = os.path.join("{}_segmentations".format(self.dataset_root), self.category, phase)
        self.samples = self.load_dataset()
        self.feat_load = False

    def load_dataset(self):
        samples = []
        defect_types = os.listdir(self.img_path)
        for defect_type in defect_types:
            if defect_type in ['good', 'ok']:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")

                for img_path in img_paths:
                    img_name = os.path.basename(img_path)
                    seg_path = os.path.join(self.seg_root, defect_type, img_name)
                    samples.append([img_path, seg_path, '', defect_type, 0])

            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")

                for img_path in img_paths:
                    img_name = os.path.basename(img_path)
                    seg_path = os.path.join(self.seg_root, defect_type, img_name)
                    gt_path = os.path.join(self.gt_path, defect_type, img_name[:img_name.rfind('.')], '000.png')
                    samples.append([img_path, seg_path, gt_path, defect_type, 1])
        return samples


    def extract_feats(self, encoder, target_layers, feat_stride, feat_dim, device, use_aggregate_features = True, use_cache = True):

        if use_aggregate_features and use_cache:
            cache_root = os.path.join("{}_cached-{}-{}-{}-{}".format(self.dataset_root, encoder.name, str(target_layers), feat_stride, feat_dim),
                                      self.category, self.phase)

        desc = 'Extract Aggregate Features: {}-{}'.format(self.category, self.phase) if use_aggregate_features \
            else 'Extract Features: {}-{}'.format(self.category, self.phase)

        for idx, sample in tqdm(enumerate(self.samples), total= len(self.samples), ncols=110,
                                desc=desc):
            image_path = sample[0]
            defect_type = sample[3]
            if use_aggregate_features and use_cache:
                image_name = os.path.basename(image_path)
                cache_name = image_name[:image_name.rfind('.')] + ".pkl"
                if os.path.exists(os.path.join(cache_root, defect_type, cache_name)):
                    feat = torch.load(os.path.join(cache_root, defect_type, cache_name)).cpu()
                else:
                    pil_img = Image.open(sample[0]).convert("RGB").resize((self.image_size, self.image_size))
                    feat = aggregate_features(encoder,
                                              target_layers,
                                              feat_stride, feat_dim,
                                              self.transform, pil_img, device, padding=False)
                    os.makedirs(os.path.join(cache_root, defect_type), exist_ok=True)
                    torch.save(feat.cpu(), os.path.join(cache_root, defect_type, cache_name))

            else:
                pil_img = Image.open(sample[0]).convert("RGB").resize((self.image_size, self.image_size))
                if use_aggregate_features:
                    feat = aggregate_features(encoder,
                                              target_layers,
                                              feat_stride, feat_dim,
                                              self.transform, pil_img, device, padding=False)
                else:
                    H, W = pil_img.height, pil_img.width
                    image = self.transform(pil_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        en_list = encoder.get_intermediate_layers(image, n=target_layers, norm=False)
                        tok = multi_scale_feat_fusion(en_list, feat_dim)[0]
                        P = tok.size(0)
                        w_p = int(round(math.sqrt(P)))
                        h_p = P // w_p
                        feat = tok.movedim(1, 0).reshape(feat_dim, h_p, w_p)
                        feat = F.interpolate(feat.unsqueeze(0), size=(H // feat_stride, W // feat_stride), mode='bilinear')[0]

            self.samples[idx].append(feat.cpu())
        self.feat_load = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        assert self.feat_load, 'Please first use the `extract_feats` function to extract pretrained features.'
        img_path, seg_path, gt_path, defect_type, label, feat = self.samples[idx]

        image = Image.open(img_path).convert('RGB').resize((self.image_size, self.image_size))

        data = {
                "feat": feat,
                "defect_type": defect_type,
                "image_path": img_path,
                "label": label
         }

        image_tensor = self.transform(image)
        seg_map = Image.open(seg_path).resize((self.image_size, self.image_size), resample = Image.Resampling.NEAREST)
        seg_map = torch.from_numpy(np.array(seg_map))

        if label == 0:
            gt = torch.zeros([1, image_tensor.size()[-2], image_tensor.size()[-2]])
        else:
            gt = Image.open(gt_path)
            gt = self.gt_transform(gt)

        data.update(
           {
            "image": image_tensor,
            "seg_map": seg_map,
            "gt": gt,
           }
        )
        return data



class ADDatasetWithSeg(torch.utils.data.Dataset):

    def __init__(self, samples, root, image_size, transform, gt_transform, phase):
        assert phase in ['train', 'validation', 'test']

        self.root = root
        self.dataset_root, self.category = os.path.split(self.root)
        self.image_size = image_size
        self.phase = phase
        self.transform = transform
        self.gt_transform = gt_transform
        self.samples = samples
        self.feat_load = False

    def extract_feats(self, encoder, target_layers, feat_stride, feat_dim, device, use_aggregate_features = True, use_cache = True):

        if use_aggregate_features and use_cache:
            cache_root = os.path.join("{}_cached-{}-{}-{}-{}".format(self.dataset_root, encoder.name, str(target_layers), feat_stride, feat_dim),
                                      self.category, self.phase)

        desc = 'Extract Aggregate Features: {}-{}'.format(self.category, self.phase) if use_aggregate_features \
            else 'Extract Features: {}-{}'.format(self.category, self.phase)

        for idx, sample in tqdm(enumerate(self.samples), total= len(self.samples), ncols=110,
                                desc=desc):

            image_path = sample[0]
            defect_type = sample[3]
            if use_aggregate_features and use_cache:
                image_name = os.path.basename(image_path)
                cache_name = image_name[:image_name.rfind('.')] + ".pkl"
                if os.path.exists(os.path.join(cache_root, defect_type, cache_name)):
                    feat = torch.load(os.path.join(cache_root, defect_type, cache_name)).cpu()
                else:
                    pil_img = Image.open(sample[0]).convert("RGB").resize((self.image_size, self.image_size))
                    feat = aggregate_features(encoder,
                                              target_layers,
                                              feat_stride, feat_dim,
                                              self.transform, pil_img, device)

                    os.makedirs(os.path.join(cache_root, defect_type), exist_ok=True)
                    torch.save(feat.cpu(), os.path.join(cache_root, defect_type, cache_name))
            else:
                pil_img = Image.open(sample[0]).convert("RGB").resize((self.image_size, self.image_size))
                if use_aggregate_features:
                    feat = aggregate_features(encoder,
                                              target_layers,
                                              feat_stride, feat_dim,
                                              self.transform, pil_img, device)
                else:
                    H, W = pil_img.height, pil_img.width
                    image = self.transform(pil_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        en_list = encoder.get_intermediate_layers(image, n=target_layers, norm=False)
                        tok = multi_scale_feat_fusion(en_list, feat_dim)[0]
                        P = tok.size(0)
                        w_p = int(round(math.sqrt(P)))
                        h_p = P // w_p
                        feat = tok.movedim(1, 0).reshape(feat_dim, h_p, w_p)
                        feat = F.interpolate(feat.unsqueeze(0), size=(H // feat_stride, W // feat_stride), mode='bilinear')[0]

            self.samples[idx].append(feat.cpu())

        self.feat_load = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        assert self.feat_load, 'Please first use the `extract_feats` function to extract pretrained features.'
        img_path, seg_path, gt_path, defect_type, label, feat = self.samples[idx]

        image = Image.open(img_path).convert('RGB').resize((self.image_size, self.image_size))

        data = {
                "feat": feat,
                "defect_type": defect_type,
                "image_path": img_path,
                "label": label
         }

        image_tensor = self.transform(image)

        if os.path.exists(seg_path):
            seg_map = Image.open(seg_path).resize((self.image_size, self.image_size), resample = Image.Resampling.NEAREST)
            seg_map = torch.from_numpy(np.array(seg_map))
        else:
            seg_map = torch.zeros((self.image_size, self.image_size))

        if label == 0:
            gt = torch.zeros([1, image_tensor.size()[-2], image_tensor.size()[-2]])
        else:
            gt = Image.open(gt_path)
            gt = self.gt_transform(gt)

        data.update(
           {
            "image": image_tensor,
            "seg_map": seg_map,
            "gt": gt,
           }
        )
        return data