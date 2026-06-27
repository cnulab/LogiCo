import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from utils.losses import global_cosine_hm_adaptive
from torchvision.ops.focal_loss import sigmoid_focal_loss
from tqdm import tqdm
from .unet import AutoencoderModel, UnetModel
from .simple_unet import UNet
import math


class LogiCo(nn.Module):
    def __init__(self,
                 encoder,
                 comp_num,
                 encoder_feat_dim,

                 CRN_num_res_blocks,
                 CRN_channel,
                 CRN_channel_mult,
                 CRN_attention_mult,

                 SRN_num_res_blocks,
                 SRN_channel,
                 SRN_channel_mult,
                 SRN_attention_mult,

                 SMD_size,
                 SMD_channel,
                 SMD_channel_mult,
                 SMD_num_res_blocks,
                 device ):

        super().__init__()

        self.encoder = encoder
        self.comp_num = comp_num
        self.encoder_feat_dim = encoder_feat_dim

        self.CRN = AutoencoderModel(
                            in_channels=encoder_feat_dim,
                            out_channels=encoder_feat_dim,
                            model_channels=CRN_channel,
                            num_res_blocks=CRN_num_res_blocks,
                            channel_mult=CRN_channel_mult,
                            attention_mult=CRN_attention_mult)

        self.SRN = UnetModel(
                            in_channels=encoder_feat_dim,
                            out_channels=encoder_feat_dim,
                            model_channels=SRN_channel,
                            num_res_blocks=SRN_num_res_blocks,
                            channel_mult=SRN_channel_mult,
                            attention_mult=SRN_attention_mult)

        self.init_weights(self.CRN)
        self.init_weights(self.SRN)

        self.CRN_mean, self.CRN_std = None, None
        self.SRN_mean, self.SRN_std = None, None

        if self.comp_num != 1:
            self.SMD = UNet(
                comp_num,
                final_channels = 1,
                n_channels = SMD_channel,
                ch_mults = SMD_channel_mult,
                n_blocks = SMD_num_res_blocks,
            )
            self.SMD_size = SMD_size
            self.init_weights(self.SMD)
            self.SMD_mean, self.SMD_std = None, None

        self.gaussian_kernel = self.get_gaussian_kernel()
        self.device = device
        self.to(device)

    def init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def get_gaussian_kernel(self, kernel_size=3, sigma=2, channels=1):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )

        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                          groups=channels,
                                          bias=False, padding=kernel_size // 2)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        return gaussian_filter


    @torch.no_grad()
    def get_comp_prototypes(self, feats, seg_maps):
        B, C, H, W = feats.shape
        seg_maps = seg_maps.long()
        comp_prototypes = torch.zeros((B, self.comp_num, C)).to(device=self.device).float()
        for idx, (feat, seg_map) in enumerate(zip(feats, seg_maps)):
            comp_ids = torch.unique(seg_map)
            for comp_id in comp_ids:
                comp_prototypes[idx][comp_id, :] = torch.mean(feat[:, seg_map == comp_id], dim=-1)
        return comp_prototypes


    @torch.no_grad()
    def resize_seg_map(self, seg_maps, size):
        if isinstance(size, int):
            size = (size, size)
        return F.interpolate(seg_maps.unsqueeze(1), size = size, mode='nearest').squeeze(1)


    @torch.no_grad()
    def lookup(self, seg_maps, comp_prototypes):
        seg_maps = seg_maps.long()
        return torch.stack([comp_prototype[seg_map].permute(2, 0, 1).contiguous()
                for seg_map, comp_prototype in zip(seg_maps, comp_prototypes)])


    @torch.no_grad()
    def get_one_hot_encoding(self, seg_map):
        seg_map = seg_map.long().unsqueeze(1)
        onehot_img = torch.zeros_like(seg_map).repeat(1, self.comp_num, 1, 1)
        onehot_seg_map = onehot_img.scatter(1, seg_map.long(), 1)
        return onehot_seg_map.float()


    @torch.no_grad()
    def cal_anomaly_maps(self, fs, ft):
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        return a_map

    @torch.no_grad()
    def anomaly_map_fusion(self, CRN_anomaly_map, SRN_anomaly_map, SMD_pred_mask, output_size):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        CRN_anomaly_map = F.interpolate(CRN_anomaly_map, size=output_size, mode='bilinear', align_corners=True)
        SRN_anomaly_map = F.interpolate(SRN_anomaly_map, size=output_size, mode='bilinear', align_corners=True)

        CRN_anomaly_map = (CRN_anomaly_map - self.CRN_mean) / self.CRN_std
        SRN_anomaly_map = (SRN_anomaly_map - self.SRN_mean) / self.SRN_std

        if SMD_pred_mask is not None:
            SMD_pred_mask = F.interpolate(SMD_pred_mask, size=output_size, mode='bilinear', align_corners=True)
            SMD_pred_mask = (SMD_pred_mask - self.SMD_mean) / self.SMD_std
            log_anomaly_map = torch.where(CRN_anomaly_map > SMD_pred_mask, CRN_anomaly_map, SMD_pred_mask)
            anomaly_map = torch.where(log_anomaly_map > SRN_anomaly_map, log_anomaly_map, SRN_anomaly_map)
        else:
            anomaly_map = torch.where(CRN_anomaly_map > SRN_anomaly_map, CRN_anomaly_map, SRN_anomaly_map)

        return self.gaussian_kernel(anomaly_map)

        
    def save_ckpt(self, path):
        state_dict = {
            "CRN": self.CRN.state_dict(),
            "SRN": self.SRN.state_dict(),
            "CRN_mean": self.CRN_mean,
            "CRN_std": self.CRN_std,
            "SRN_mean": self.SRN_mean,
            "SRN_std": self.SRN_std,
        }

        if self.comp_num != 1:
            state_dict.update({
                "SMD": self.SMD.state_dict(),
                "SMD_mean": self.SMD_mean,
                "SMD_std": self.SMD_std,
            })

        torch.save(state_dict, path)


    def load_ckpt(self, path):
        state_dict = torch.load(path)
        self.CRN.load_state_dict(state_dict['CRN'])
        self.SRN.load_state_dict(state_dict['SRN'])

        self.CRN_mean = state_dict['CRN_mean']
        self.CRN_std = state_dict['CRN_std']

        self.SRN_mean = state_dict['SRN_mean']
        self.SRN_std = state_dict['SRN_std']

        if self.comp_num != 1:
            self.SMD.load_state_dict(state_dict['SMD'])
            self.SMD_mean = state_dict['SMD_mean']
            self.SMD_std = state_dict['SMD_std']

    def get_trainable_params(self):
        if self.comp_num != 1:
            return list(self.CRN.parameters()) + list(self.SRN.parameters()) + list(self.SMD.parameters())
        else:
            return list(self.CRN.parameters()) + list(self.SRN.parameters())


    @torch.no_grad()
    def get_scores_mean_and_std(self, dataloader):

        CRN_anomaly_maps = []
        SRN_anomaly_maps = []
        SMD_anomaly_maps = []

        for data in tqdm(dataloader, desc='Score Normalization', total=len(dataloader)):

            feats = data['feat'].to(self.device)
            seg_map = data['seg_map'].to(self.device).float()
            B, C, H, W = feats.shape

            resize_seg_map = self.resize_seg_map(seg_map, size=(H, W))
            comp_prototypes = self.get_comp_prototypes(feats, resize_seg_map)
            dis_feats = self.lookup(resize_seg_map, comp_prototypes)

            comp_recon_feats, hs = self.CRN(dis_feats)
            CRN_anomaly_map = self.cal_anomaly_maps(feats, comp_recon_feats)
            CRN_anomaly_maps.append(CRN_anomaly_map.cpu())

            str_recon_feats = self.SRN(feats, hs[::-1])
            str_anomaly_map = self.cal_anomaly_maps(feats, str_recon_feats)
            SRN_anomaly_maps.append(str_anomaly_map.cpu())

            if self.comp_num != 1:
                dis_input_seg_map = self.resize_seg_map(seg_map, size=self.SMD_size)
                onehot_seg_map = self.get_one_hot_encoding(dis_input_seg_map)
                SMD_pred_mask = nn.Sigmoid()(self.SMD(onehot_seg_map))
                SMD_anomaly_maps.append(SMD_pred_mask.cpu())

        CRN_anomaly_maps = torch.cat(CRN_anomaly_maps).view(-1)
        SRN_anomaly_maps = torch.cat(SRN_anomaly_maps).view(-1)

        CRN_anomaly_maps, _ = torch.topk(CRN_anomaly_maps, int(CRN_anomaly_maps.numel() * 0.2))
        SRN_anomaly_maps, _ = torch.topk(SRN_anomaly_maps, int(SRN_anomaly_maps.numel() * 0.2))

        self.CRN_mean, self.CRN_std = torch.mean(CRN_anomaly_maps), torch.std(CRN_anomaly_maps)
        self.SRN_mean, self.SRN_std = torch.mean(SRN_anomaly_maps), torch.std(SRN_anomaly_maps)

        if self.comp_num != 1:
            SMD_anomaly_maps = torch.cat(SMD_anomaly_maps).view(-1)
            SMD_anomaly_maps, _ = torch.topk(SMD_anomaly_maps, int(SMD_anomaly_maps.numel() * 0.2))
            self.SMD_mean, self.SMD_std = torch.mean(SMD_anomaly_maps), torch.std(SMD_anomaly_maps)


    def get_losses(self, data):

        feats = data['feat'].to(self.device)
        B, C, H, W = feats.shape
        seg_map = data['seg_map'].to(self.device).float()
        anomaly_seg_map = data['anomaly_seg_map'].to(self.device).float()

        resize_seg_map = self.resize_seg_map(seg_map, size=(H, W))
        resize_anomaly_seg_map = self.resize_seg_map(anomaly_seg_map, size=(H, W))

        comp_prototypes = self.get_comp_prototypes(feats, resize_seg_map)
        comp_anomaly_feats = self.lookup(resize_anomaly_seg_map, comp_prototypes)
        comp_recon_feats, hs = self.CRN(comp_anomaly_feats)
        CRN_recon_loss = global_cosine_hm_adaptive(feats, comp_recon_feats, y=4)

        str_recon_feats = self.SRN(feats, hs[::-1], training=True)
        SRN_recon_loss = global_cosine_hm_adaptive(feats, str_recon_feats, y=4)

        if self.comp_num != 1:
            dis_input_seg_map = self.resize_seg_map(anomaly_seg_map, size = self.SMD_size)
            onehot_seg_map = self.get_one_hot_encoding(dis_input_seg_map)
            pred_mask = self.SMD(onehot_seg_map).squeeze(1)
            masks = self.resize_seg_map(data['masks'], size = self.SMD_size).to(self.device)
            SMD_loss = 5 * sigmoid_focal_loss(pred_mask, masks).mean() + F.l1_loss(nn.Sigmoid()(pred_mask), masks)
        else:
            SMD_loss = torch.zeros(1).to(self.device)

        return CRN_recon_loss, SRN_recon_loss, SMD_loss


    def forward(self, data):

        feats = data['feat'].to(self.device)
        seg_map = data['seg_map'].to(self.device).float()

        B, C, H, W = feats.shape
        resize_seg_map = self.resize_seg_map(seg_map, size=(H, W))
        comp_prototypes = self.get_comp_prototypes(feats, resize_seg_map)
        comp_feats = self.lookup(resize_seg_map, comp_prototypes)

        comp_recon_feats, hs = self.CRN(comp_feats)
        CRN_anomaly_map = self.cal_anomaly_maps(feats, comp_recon_feats)

        str_recon_feats = self.SRN(feats, hs[::-1])
        SRN_anomaly_map = self.cal_anomaly_maps(feats, str_recon_feats)

        if self.comp_num != 1:
            dis_input_seg_map = self.resize_seg_map(seg_map, size=self.SMD_size)
            onehot_seg_map = self.get_one_hot_encoding(dis_input_seg_map)
            SMD_pred_mask = nn.Sigmoid()(self.SMD(onehot_seg_map))
        else:
            SMD_pred_mask = None

        return self.anomaly_map_fusion(CRN_anomaly_map, SRN_anomaly_map, SMD_pred_mask, data['image'].shape[-1])
