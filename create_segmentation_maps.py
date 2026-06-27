import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
import sys
from PIL import Image
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
sys.path.append("../")
import torch
from dinov3.data.transforms import make_classification_eval_transform
import torch.nn.functional as F
import numpy as np
import random
import math
from tqdm import tqdm
import glob
import cv2
import argparse
import yaml
from easydict import EasyDict
import os
from typing import Sequence, List, Tuple
import re
import warnings
warnings.filterwarnings('ignore')


def gen_crops(h: int, w: int) -> List[Tuple[int, int, int, int]]:

    def extract_starts(image_size, patch_size, stride):
        if image_size <= patch_size:
            starts = [0, ]
        else:
            starts = list(range(0, image_size, stride))
            for i in range(len(starts)):
                if starts[i] + patch_size > image_size:
                    starts[i] = image_size - patch_size
            starts = sorted(set(starts), key=starts.index)
        return starts

    AREA_FRACS = [0.1, 0.5, 1.0]
    STRIDE_RATIO = 0.5
    crops = []

    for af in AREA_FRACS:
        side = max(16, int(round(math.sqrt(af) * min(h, w))))  # >= 16
        stride = max(8, int(round(side * STRIDE_RATIO)))  # >= 8

        y_starts_list = extract_starts(h, side, stride)
        x_starts_list = extract_starts(w, side, stride)
        for x in x_starts_list:
            for y in y_starts_list:
                crops.append([x, y, x+side, y+side])

    return crops


@torch.no_grad()
def aggregate_features(model, preprocess, pil_img: Image.Image, batch_size):
    H, W = pil_img.height, pil_img.width

    post_fmap, post_hits = None, torch.zeros((H, W), device=device)
    per_fmap, per_hits = None, torch.zeros((H, W), device=device)

    crop_images = []
    points = []

    for (x0, y0, x1, y1) in gen_crops(H, W):
        crop = pil_img.crop((x0, y0, x1, y1))
        crop_t = preprocess(crop).unsqueeze(0).to(device)
        crop_images.append(crop_t)
        points.append([x0, y0, x1, y1])
    crop_images = torch.cat(crop_images)

    post_toks = []
    per_toks = []

    for chunk_crop_images in torch.split(crop_images, split_size_or_sections = batch_size if batch_size != -1 else crop_images.shape[0], dim=0):
        chunk_post_toks, chunk_per_toks = encode_patches(model, chunk_crop_images)
        post_toks.append(chunk_post_toks)
        per_toks.append(chunk_per_toks)

    post_toks = torch.cat(post_toks)
    per_toks = torch.cat(per_toks)

    for point, post_tok, per_tok in zip(points, post_toks, per_toks):

        if post_fmap is None:
            post_fmap = torch.zeros((post_tok.size(1), H, W), device=device)

        if per_fmap is None:
            per_fmap = torch.zeros((per_tok.size(1), H, W), device=device)

        x0, y0, x1, y1 = point

        P = post_tok.size(0)

        w_p = int(round(math.sqrt(P)))
        if P % w_p != 0:
            for w_p in range(w_p, 0, -1):
                if P % w_p == 0:
                    break
        h_p = P // w_p

        post_grid = post_tok.movedim(1, 0).reshape(post_fmap.size(0), h_p, w_p)
        post_grid = \
        F.interpolate(post_grid.unsqueeze(0), size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)[0]
        post_fmap[:, y0:y1, x0:x1] += post_grid
        post_hits[y0:y1, x0:x1] += 1

        P = per_tok.size(0)

        w_p = int(round(math.sqrt(P)))
        if P % w_p != 0:
            for w_p in range(w_p, 0, -1):
                if P % w_p == 0:
                    break
        h_p = P // w_p

        per_grid = per_tok.movedim(1, 0).reshape(per_fmap.size(0), h_p, w_p)
        per_grid = F.interpolate(per_grid.unsqueeze(0), size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)[
            0]
        per_fmap[:, y0:y1, x0:x1] += per_grid
        per_hits[y0:y1, x0:x1] += 1

    return post_fmap / post_hits.clamp(min=1), per_fmap / per_hits.clamp(min=1)



@torch.no_grad()
def build_text_embeddings(model, tokenizer, class_names: Sequence[str], prompt_templates) -> torch.Tensor:
    prompts, owners = [], []
    for c_idx, name in enumerate(class_names):
        for tpl in prompt_templates:
            prompts.append(tpl.format(name))
            owners.append(c_idx)
    toks = tokenizer.tokenize(prompts).to(device)
    embs = model.encode_text(toks)[:, 1024:]
    C, D = len(class_names), embs.size(1)
    agg = torch.zeros(C, D, device=embs.device)
    cnt = torch.zeros(C, device=embs.device)
    for i, c in enumerate(owners):
        agg[c] += embs[i]
        cnt[c] += 1
    return F.normalize(agg / cnt.unsqueeze(1), p=2, dim=1)  # [C, D]


def encode_patches(model, img_tensor: torch.Tensor):
    with torch.no_grad():
        image_class_tokens, image_patch_tokens, backbone_patch_tokens = model.encode_image_with_patch_tokens(img_tensor)
    return image_patch_tokens, backbone_patch_tokens

def get_connected_areas(mask):
    mask = mask.astype(np.uint8)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = []
    sizes = []
    for id, (x, y, h, w, s) in enumerate(stats[1:]):
        area = np.zeros_like(mask)
        area[labels == (id + 1)] = 1
        areas.append(area)
        sizes.append(s)
    return areas, sizes


def get_neighborhood_id(target_mask, seg_map, r=4):
    kernel = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    dilated = cv2.dilate(target_mask, kernel, iterations=1)
    neighborhoods = seg_map[np.logical_xor(target_mask, dilated)]
    values, counts = np.unique(neighborhoods, return_counts=True)
    return values[np.argmax(counts)]


def remove_small_holes(seg_map, threshold=1e-3):
    H, W = seg_map.shape
    seg_map = seg_map.copy().astype(np.uint8)
    id_list = np.unique(seg_map)

    for id in id_list:
        mask = np.zeros_like(seg_map).astype(np.uint8)
        mask[seg_map == id] = 1
        areas, sizes = get_connected_areas(mask)
        for area, size in zip(areas, sizes):
            if size < int(H * W * threshold):
                replace_id = get_neighborhood_id(area, seg_map)
                seg_map[area.astype(bool)] = replace_id
    return seg_map


@torch.no_grad()
def main(args):
    random.seed(0)
    torch.manual_seed(0)

    category = args.category
    dataset = args.dataset

    with open(os.path.join(args.config_root, "segmentations", dataset, '{}.yaml'.format(category))) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(pretrained=False,
                                                            vision_backbone_pretrained_path=os.path.join(args.ckpt_root,
                                                            'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'),
                                                            txt_backbone_pretrained_path=os.path.join(args.ckpt_root,
                                                            'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth'))


    model = model.to(device)
    model.eval()

    preprocess = make_classification_eval_transform(resize_size=config.image_size, crop_size=config.image_size)

    text_emb = build_text_embeddings(model, tokenizer, config['prompts'], config['prompt_templates'])

    cluster_samples = random.sample(
        glob.glob(f"{'{}/{}/{}/train/'.format(args.data_root, args.dataset, category)}/*/*"), config.cluster_sample_num)

    comp_names = config['prompts']
    if len(comp_names) == 1:
        print("Single component images do not require running this program.")
        exit(0)

    feat_memory_bank = {i: [] for i in range(len(comp_names))}

    for file in tqdm(cluster_samples, desc='Starting Clustering: {}-{}'.format(args.dataset, category)):

        pil_img = Image.open(file).convert("RGB").resize((config.image_size, config.image_size))
        fmap_post, fmap_per = aggregate_features(model, preprocess, pil_img, args.batch_size)

        fmap_post = fmap_post.permute(1, 2, 0).view((config.image_size * config.image_size, -1))
        fmap_post = F.normalize(fmap_post.float(), p=2, dim=1)
        pred_map = (fmap_post @ text_emb.T)

        pred_map = pred_map.permute(1, 0).view((-1, config.image_size, config.image_size)).cpu().numpy()
        pred_map_idx = pred_map.argmax(0)

        for idx in range(len(comp_names)):
            mask_idx = (pred_map_idx == idx)
            if np.mean(mask_idx) >= config.area_threshold:
                score = pred_map[idx][mask_idx]
                feat = fmap_per[:, mask_idx]
                n = int(score.shape[0] * config.topk)
                indexes = np.argpartition(score, -n)[-n:]
                feat_memory_bank[idx].append(feat[:, indexes])

    for idx in feat_memory_bank:
        feat_memory_bank[idx] = torch.mean(torch.concatenate(feat_memory_bank[idx], dim=1), dim=1)

    centers = torch.stack([feat_memory_bank[key] for key in feat_memory_bank]).to(device)

    splits = ['train', 'validation', 'test'] if args.dataset == 'mvtec_loco' else ['train', 'test']

    for split in splits:
        src_root = os.path.join(args.data_root, args.dataset, category, split)

        files = glob.glob(f"{src_root}/*/*")
        files.sort()

        save_root = os.path.join(args.data_root, "{}_segmentations".format(args.dataset), category, split)
        os.makedirs(save_root, exist_ok=True)

        for file in tqdm(files, ncols=110, desc='Segmentation: {}-{}-{}'.format(args.dataset, category, split)):
            pil_img = Image.open(file).convert("RGB").resize((config.image_size, config.image_size))
            _, fmap_per = aggregate_features(model, preprocess, pil_img, args.batch_size)  # (C,H,W)

            fmap_per = fmap_per.reshape((-1, config.image_size * config.image_size)).transpose(1, 0)
            seg_map = torch.argmin(torch.cdist(fmap_per, centers), dim=-1).reshape((config.image_size, config.image_size)).cpu().numpy()
            seg_map = remove_small_holes(seg_map, threshold = config.remove_small_holes_threshold)

            file_root, file_name = os.path.split(file)
            file_root, class_name = os.path.split(file_root)

            seg_example_root = os.path.join(save_root, class_name)
            os.makedirs(seg_example_root, exist_ok=True)

            pi = Image.fromarray(seg_map.astype(np.uint8), 'P')
            pi.putpalette(config['palette'])
            pi.save(os.path.join(seg_example_root, re.sub('jpg', 'png', file_name, flags=re.IGNORECASE)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root', type=str, default='data', help='Dataset directory.')
    parser.add_argument('--ckpt_root', type=str, default='ckpt', help='Checkpoint directory.')
    parser.add_argument('--dataset', type=str, default='mvtec_loco', choices=['mvtec_loco', 'mvtec', 'visa', 'realiad'])
    parser.add_argument('--category', type=str, default='breakfast_box', help='Image category.')
    parser.add_argument('--config_root', type=str, default='configs', help='Configuration file directory.')
    parser.add_argument('--batch_size', type=int, default=-1)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
