import torch
import os
import warnings
import argparse
from utils.utils import setup_seed, get_logger, get_dinov3_weight_name, anomaly_map_normalization
from dinov3.hub.backbones import load_dinov3_model
from utils.dataset import get_data_transforms, load_anomaly_detection_dataset, LOCODatasetWithSeg, ADDatasetWithSeg
import yaml
from easydict import EasyDict
from models import LogiCo
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from PIL import Image
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils.metrics import compute_imagewise_log_metrics, compute_pixelwise_metrics, \
    compute_imagewise_metrics, compute_pro, compute_pixelwise_sPRO_metrics, metrics2str

warnings.filterwarnings("ignore")

def main(args):

    setup_seed(1)

    data_path = os.path.join(args.data_root, args.dataset, args.category)

    with open(os.path.join(args.config_root, 'train', '{}.yaml'.format(args.dataset))) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    with open(os.path.join(args.config_root, "segmentations", args.dataset, '{}.yaml'.format(args.category))) as f:
        config_segmentations = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.comp_names = config_segmentations['prompts']

    data_transform, gt_transform = get_data_transforms(config.image_size)

    if args.dataset == 'mvtec_loco':
        test_data = LOCODatasetWithSeg(root=data_path,  image_size=config.image_size,
                                       transform=data_transform, gt_transform=gt_transform,
                                       phase="test")
    else:
        seg_path = os.path.join(args.data_root, "{}_segmentations".format(args.dataset), args.category)
        test_samples = load_anomaly_detection_dataset(args.dataset, data_path, seg_path, split='test')
        test_data = ADDatasetWithSeg(root=data_path, samples=test_samples,
                                     image_size=config.image_size, transform=data_transform, gt_transform=gt_transform,
                                     phase="test")

    encoder = load_dinov3_model(config.encoder, pretrained_weight_path=os.path.join(args.ckpt_root, get_dinov3_weight_name(config.encoder)))
    encoder.name = config.encoder
    encoder.cuda()
    encoder.eval()

    test_data.extract_feats(encoder, config.encoder_target_layers, config.encoder_feat_stride, config.encoder_feat_dim, device=device,
                            use_aggregate_features = config.use_aggregate_features, use_cache = True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = LogiCo(**config.model,
                   encoder=encoder,
                   comp_num=len(config.comp_names),
                   encoder_feat_dim= config.encoder_feat_dim,
                   device = device)
    
    model.load_ckpt(os.path.join(args.save_dir,  args.dataset, args.category, args.ckpt_file_name))
    model.eval()
    
    img_paths = []
    gt_image_list = []
    pr_image_list = []
    gt_pixel_list = []
    pr_pixel_list = []
    
    with torch.no_grad():
    
        for data in tqdm(test_dataloader, total=len(test_dataloader), desc = 'Testing'):
            anomaly_map = model(data)
            gt = data['gt']
    
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
    
            anomaly_map = F.interpolate(anomaly_map, size=config.resize_mask, mode='bilinear', align_corners=False)
            gt = F.interpolate(gt, size=config.resize_mask, mode='nearest')
    
            gt_image_list.append(data['label'])
            gt_pixel_list.append(gt)
            pr_pixel_list.append(anomaly_map)
    
            if config.max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * config.max_ratio)]
                sp_score = sp_score.mean(dim=1)
    
            pr_image_list.append(sp_score)
            img_paths.extend(data['image_path'])
    
        gt_image_list = torch.cat(gt_image_list).flatten().cpu().numpy()
        pr_image_list = torch.cat(pr_image_list).flatten().cpu().numpy()
        gt_pixel_list = torch.cat(gt_pixel_list).squeeze(1).cpu().numpy()
        pr_pixel_list = torch.cat(pr_pixel_list).squeeze(1).cpu().numpy()
    
    if args.dataset == 'mvtec_loco':
        results = compute_imagewise_log_metrics(pr_image_list = pr_image_list, gt_image_list = gt_image_list, img_paths = img_paths)
        # results.update(compute_pixelwise_sPRO_metrics(pr_pixel_list = pr_pixel_list, img_paths = img_paths, data_root= data_path ))
    else:
        results = compute_imagewise_metrics(pr_image_list=pr_image_list, gt_image_list=gt_image_list)
        results.update(compute_pixelwise_metrics(pr_pixel_list=pr_pixel_list, gt_pixel_list=gt_pixel_list, metrics_name=['pAUC', 'pAP']))
        results.update(compute_pro(pr_pixel_list=pr_pixel_list, gt_pixel_list=gt_pixel_list))
    
    print_fn('{}: {}'.format(args.category, metrics2str(results)))
    
    if config.vis_size is not None:
    
        pr_pixel_list = anomaly_map_normalization(pr_pixel_list)
    
        for index in tqdm(range(len(img_paths)), total=len(img_paths), desc='Saving vis results:'):
            img_path = img_paths[index]
            gt = gt_pixel_list[index]
            pr = pr_pixel_list[index]
    
            image = np.array(Image.open(img_path).convert('RGB').resize((config.vis_size, config.vis_size)))
            pr = cv2.resize(pr, dsize = (config.vis_size, config.vis_size), interpolation = cv2.INTER_LINEAR)
            gt = cv2.resize(gt, dsize = (config.vis_size, config.vis_size), interpolation = cv2.INTER_NEAREST)
    
            heat = show_cam_on_image(image / 255, pr, use_rgb=True)
            save_image = np.concatenate([image, heat, np.repeat((gt * 255)[..., None], 3, -1)], axis=1).astype(np.uint8)
    
            img_root, image_name = os.path.split(img_path)
            _, defect_type = os.path.split(img_root)
            vis_root = os.path.join(args.save_dir, args.dataset, args.category, 'vis', defect_type)
            os.makedirs(vis_root, exist_ok=True)
            Image.fromarray(save_image).save(os.path.join(vis_root, image_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data_root', type=str, default='data', help='Dataset directory.')
    parser.add_argument('--ckpt_root', type=str, default='ckpt', help='Checkpoint directory.')
    parser.add_argument('--config_root', type=str, default='configs', help='Configuration file directory.')
    parser.add_argument('--dataset', type=str, default=r'mvtec_loco', choices=['mvtec_loco', 'mvtec', 'visa', 'realiad'])
    parser.add_argument('--category', type=str, default='breakfast_box', help='Image category.')
    parser.add_argument('--save_dir', type=str, default='saved_results')
    parser.add_argument('--ckpt_file_name', type=str, default='LogiCo_best_ckpt.pkl')
    parser.add_argument('--batch_size', type=int, default = 16)
    args = parser.parse_args()

    logger = get_logger(os.path.join(args.save_dir, args.dataset, args.category))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    main(args)
