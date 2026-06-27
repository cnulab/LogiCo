import torch
import os
import warnings
import argparse
from utils.utils import setup_seed, get_logger, get_dinov3_weight_name, WarmCosineScheduler
from dinov3.hub.backbones import load_dinov3_model
from utils.dataset import get_data_transforms, load_anomaly_detection_dataset, LOCODatasetWithSeg, ADDatasetWithSeg
import yaml
from easydict import EasyDict
from models import LogiCo
from utils.syn import CombSynthesizer
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import compute_imagewise_log_metrics, compute_imagewise_metrics, compute_pixelwise_metrics, metrics2str

warnings.filterwarnings("ignore")


def main(args):

    setup_seed(args.seed)

    data_path = os.path.join(args.data_root, args.dataset, args.category)

    with open(os.path.join(args.config_root, 'train', '{}.yaml'.format(args.dataset))) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    with open(os.path.join(args.config_root, "segmentations", args.dataset, '{}.yaml'.format(args.category))) as f:
        config_segmentations = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.comp_names = config_segmentations['prompts']

    print_fn(config)
    data_transform, gt_transform = get_data_transforms(config.image_size)

    if args.dataset == 'mvtec_loco':
        train_data = LOCODatasetWithSeg(root=data_path, image_size=config.image_size,
                                        transform=data_transform, gt_transform=gt_transform,
                                        phase="train")

        validation_data = LOCODatasetWithSeg(root=data_path,  image_size=config.image_size,
                                       transform=data_transform, gt_transform=gt_transform,
                                       phase="validation")

        test_data = LOCODatasetWithSeg(root=data_path,  image_size=config.image_size,
                                       transform=data_transform, gt_transform=gt_transform,
                                       phase="test")
    else:
        seg_path = os.path.join(args.data_root, "{}_segmentations".format(args.dataset), args.category)
        train_samples, validation_samples = load_anomaly_detection_dataset(args.dataset, data_path, seg_path, split='train',
                                        val_proportion = config.val_proportion)

        test_samples = load_anomaly_detection_dataset(args.dataset, data_path, seg_path, split='test')

        train_data = ADDatasetWithSeg(root=data_path, samples=train_samples,
                                       image_size=config.image_size, transform=data_transform, gt_transform=gt_transform,
                                        phase="train")

        validation_data = ADDatasetWithSeg(root=data_path,  samples=validation_samples,
                                        image_size=config.image_size, transform=data_transform, gt_transform=gt_transform,
                                         phase="validation")

        test_data = ADDatasetWithSeg(root=data_path,  samples=test_samples,
                                     image_size=config.image_size, transform=data_transform, gt_transform=gt_transform,
                                     phase="test")

    encoder = load_dinov3_model(config.encoder, pretrained_weight_path=os.path.join(args.ckpt_root, get_dinov3_weight_name(config.encoder)))
    encoder.name = config.encoder
    encoder.to(device)
    encoder.eval()
    
    train_data.extract_feats(encoder, config.encoder_target_layers, config.encoder_feat_stride, config.encoder_feat_dim,
                             use_aggregate_features = config.use_aggregate_features, device=device)
    test_data.extract_feats(encoder, config.encoder_target_layers, config.encoder_feat_stride, config.encoder_feat_dim,
                            use_aggregate_features = config.use_aggregate_features, device=device)
    validation_data.extract_feats(encoder, config.encoder_target_layers, config.encoder_feat_stride, config.encoder_feat_dim,
                            use_aggregate_features = config.use_aggregate_features, device=device)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last = True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = LogiCo(**config.model,
                   encoder=encoder,
                   comp_num=len(config.comp_names),
                   encoder_feat_dim= config.encoder_feat_dim,
                   device = device)

    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr = 2e-4, weight_decay = 1e-4)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-4, final_value=1e-4, total_iters = config.total_epochs * len(train_dataloader), warmup_iters=100)
    synthesizer = CombSynthesizer(**config.synthesizer, num_cls=len(config.comp_names))

    best_metrics = None

    for epoch in range(config.total_epochs):
        model.train()
        loss_list = []

        log_rocen_loss_list = []
        str_rocen_loss_list = []
        discriminator_loss_list = []

        for data in train_dataloader:

            anomaly_seg_map, masks = synthesizer(data['seg_map'])
            data['anomaly_seg_map'] = anomaly_seg_map
            data['masks'] = masks
            log_recon_loss, str_recon_loss, discriminator_loss = model.get_losses(data)
            loss = log_recon_loss + str_recon_loss + discriminator_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.CRN.parameters(), max_norm=0.1)
            nn.utils.clip_grad_norm(model.SRN.parameters(), max_norm=0.1)
            optimizer.step()

            loss_list.append(loss.item())
            log_rocen_loss_list.append(log_recon_loss.item())
            str_rocen_loss_list.append(str_recon_loss.item())
            discriminator_loss_list.append(discriminator_loss.item())
            lr_scheduler.step()

        print_fn('epoch [{}/{}], loss: {:.4f}, log-recon-loss: {:.4f}, str-recon-loss: {:.4f}, discriminator-loss: {:.4f}'.format(epoch + 1, config.total_epochs,
            np.mean(loss_list), np.mean(log_rocen_loss_list), np.mean(str_rocen_loss_list), np.mean(discriminator_loss_list)))

        if (epoch + 1) % config.evaluation_frep == 0:
            best_metrics = evaluation(args, model, validation_dataloader, test_dataloader, best_metrics, epoch + 1, max_ratio=config.max_ratio,
                                  resize_mask = config.resize_mask)


def evaluation(args, model, validation_dataloader, test_dataloader, best_metrics, epoch, max_ratio=0, resize_mask = None):

    model.eval()
    model.get_scores_mean_and_std(validation_dataloader)

    img_paths = []
    gt_image_list = []
    pr_image_list = []
    gt_pixel_list = []
    pr_pixel_list = []

    with torch.no_grad():

        for data in test_dataloader:
            anomaly_map = model(data)
            gt = data['gt']

            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')

            gt_image_list.append(data['label'])
            gt_pixel_list.append(gt)
            pr_pixel_list.append(anomaly_map)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)

            pr_image_list.append(sp_score)
            img_paths.extend(data['image_path'])

        gt_image_list = torch.cat(gt_image_list).flatten().cpu().numpy()
        pr_image_list = torch.cat(pr_image_list).flatten().cpu().numpy()
        gt_pixel_list = torch.cat(gt_pixel_list).squeeze(1).cpu().numpy()
        pr_pixel_list = torch.cat(pr_pixel_list).squeeze(1).cpu().numpy()

    if args.dataset == 'mvtec_loco':
        results = compute_imagewise_log_metrics(pr_image_list = pr_image_list, gt_image_list = gt_image_list, img_paths = img_paths)
    else:
        results = compute_imagewise_metrics(pr_image_list = pr_image_list, gt_image_list = gt_image_list)
        results.update(compute_pixelwise_metrics(pr_pixel_list = pr_pixel_list, gt_pixel_list = gt_pixel_list, metrics_name= ['pAUC']))

    print_fn('{}: {}'.format(args.category, metrics2str(results)))

    if best_metrics is None or np.mean([best_metrics[key] for key in best_metrics]) < np.mean([results[key] for key in results]):
        best_metrics = results
        model.save_ckpt(os.path.join(args.save_dir, args.dataset, args.category, 'LogiCo_best_ckpt.pkl'))

    print_fn(
        'Best Metrics: {}'.format(metrics2str(best_metrics)))

    model.save_ckpt(os.path.join(args.save_dir, args.dataset, args.category, f'LogiCo_epoch_{epoch}_ckpt.pkl'))
    return best_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root', type=str, default='data', help='Dataset directory.')
    parser.add_argument('--ckpt_root', type=str, default='ckpt', help='Checkpoint directory.')
    parser.add_argument('--config_root', type=str, default='configs', help='Configuration file directory.')
    parser.add_argument('--dataset', type=str, default=r'mvtec_loco', choices=['mvtec_loco', 'mvtec', 'visa', 'realiad'])
    parser.add_argument('--category', type=str, default='breakfast_box', help='Image category.')
    parser.add_argument('--save_dir', type=str, default='saved_results')
    parser.add_argument('--batch_size', type=int, default = 14)
    parser.add_argument('--seed', type=int, default = 42)
    args = parser.parse_args()
    logger = get_logger(os.path.join(args.save_dir, args.dataset, args.category))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    main(args)
