import torch
import numpy as np
import cv2
import random
from scipy.ndimage import map_coordinates, distance_transform_edt


class BaseSynthesizer:

    def __init__(self):
        pass

    def random_choice(self, seg_map, smallest_top = None):
        if type(seg_map) == torch.Tensor:
            seg_map = seg_map.cpu().detach().numpy()

        seg_map = seg_map.reshape(-1)

        values, counts = np.unique(seg_map, return_counts=True)

        if smallest_top is None:
            return random.choice(values.tolist())
        else:
            assert smallest_top > 0 and smallest_top < 1
            index = random.choice(np.argsort(counts)[: max(int(len(counts)*smallest_top), 2)])
            return values[index]

    def get_component_mask(self, seg_map, target_id):
        mask = np.zeros_like(seg_map).astype(np.uint8)
        mask [seg_map == target_id] = 1
        return mask

    def get_neighborhood_id(self, target_mask, seg_map, r=4):
        kernel = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
        dilated = cv2.dilate(target_mask, kernel, iterations=1)
        neighborhoods = seg_map[np.logical_xor(target_mask, dilated)]
        values, counts = np.unique(neighborhoods, return_counts=True)
        return values[np.argmax(counts)]

    def get_connected_areas(self, mask):
        mask = mask.astype(np.uint8)
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = []
        for id in range(len(stats[1:])):
            area = np.zeros_like(mask)
            area[labels == (id + 1)] = 1
            areas.append(area)
        return areas



class DeleteComponents(BaseSynthesizer):

    def __init__(self, smallest_top = None, **kwargs):
        super().__init__()
        self.smallest_top = smallest_top

    def __call__(self, seg_map):
        device = seg_map.device

        if type(seg_map) == torch.Tensor:
            seg_map = seg_map.cpu().detach().numpy()

        seg_map_copy = seg_map.copy()

        target_id = self.random_choice(seg_map_copy, smallest_top = self.smallest_top)
        target_mask = self.get_component_mask(seg_map_copy, target_id)
        replace_id = self.get_neighborhood_id(target_mask, seg_map_copy)

        seg_map_copy[target_mask.astype(bool)] = replace_id

        mask = np.zeros_like(seg_map_copy).astype(float)
        mask[seg_map_copy == replace_id] = 1

        return torch.from_numpy(seg_map_copy).to(device), torch.from_numpy(mask).to(device)


class EraseComponents(BaseSynthesizer):

    def __init__(self, erase_percent, smallest_top=None, **kwargs):
        super().__init__()
        self.erase_percent = erase_percent
        self.smallest_top = smallest_top

    def shrink_mask(self, mask, p, jitter=2):

        assert p > 0 and p < 1
        m = mask.astype(bool)

        rng = np.random.default_rng()

        d = distance_transform_edt(m)

        noise = np.zeros_like(d, dtype=float)
        if jitter and jitter > 0:
            noise[m] = rng.uniform(0.0, jitter, size=m.sum())
        d_perturbed = d + noise

        t = np.quantile(d_perturbed[m], p)
        out = m.copy()
        out[(d_perturbed <= t) & m] = False
        return np.logical_xor(m, out).astype(np.uint8)

    def __call__(self, seg_map):
        device = seg_map.device

        if type(seg_map) == torch.Tensor:
            seg_map = seg_map.cpu().detach().numpy()

        seg_map_copy = seg_map.copy()
        target_id = self.random_choice(seg_map_copy, smallest_top= self.smallest_top)
        target_mask = self.get_component_mask(seg_map_copy, target_id)
        replace_id = self.get_neighborhood_id(target_mask, seg_map_copy)
        areas = self.get_connected_areas(target_mask)

        if len(areas) > 1:
            erase_areas = random.sample(areas, max(1, int(len(areas)*self.erase_percent)))
            for area in erase_areas:
                seg_map_copy[area.astype(bool)] = replace_id
            mask = np.zeros_like(seg_map_copy).astype(float)
            mask[seg_map_copy == replace_id] = 0.5
        else:
            target_mask = self.shrink_mask(areas[0], self.erase_percent)
            seg_map_copy[target_mask.astype(bool)] = replace_id
            mask = np.zeros_like(seg_map_copy).astype(float)
            mask[seg_map_copy != seg_map] = 1

        return torch.from_numpy(seg_map_copy).to(device), torch.from_numpy(mask).to(device)



class ReplaceComponents(BaseSynthesizer):

    def __init__(self, num_cls, smallest_top = None, **kwargs):

        super().__init__()
        self.smallest_top = smallest_top
        self.num_cls = num_cls

    def mask_aug(self, M):
        rng = np.random.default_rng()
        m = M.astype(bool)
        H, W = m.shape

        scale_range = (1.0, 1.2)
        rotate_range = (-45.0, 45.0)

        sx = rng.uniform(*scale_range)
        sy = rng.uniform(*scale_range)
        rot = np.deg2rad(rng.uniform(*rotate_range))

        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

        def T(tx, ty):
            return np.array([[1, 0, tx],
                             [0, 1, ty],
                             [0, 0, 1]], dtype=np.float32)

        def R(a):
            ca, sa = np.cos(a), np.sin(a)
            return np.array([[ca, -sa, 0],
                             [sa, ca, 0],
                             [0, 0, 1]], dtype=np.float32)

        def S(sx, sy):
            return np.array([[sx, 0, 0],
                             [0, sy, 0],
                             [0, 0, 1]], dtype=np.float32)

        def _grid(H, W):
            ys = np.arange(H, dtype=np.float32)
            xs = np.arange(W, dtype=np.float32)
            return np.meshgrid(xs, ys, indexing='xy')

        C = T(-cx, -cy)
        Cinv = T(cx, cy)
        A = Cinv @ R(rot) @ S(sx, sy) @ C

        Ainv = np.linalg.inv(A)

        X, Y = _grid(H, W)
        ones = np.ones_like(X)
        pts = np.stack([X, Y, ones], axis=-1).reshape(-1, 3).T
        src = (Ainv @ pts).T.reshape(H, W, 3)
        map_x, map_y = src[..., 0], src[..., 1]

        warped = map_coordinates(m.astype(np.uint8), [map_y, map_x],
                                 order=0, mode='constant', cval=0)
        out = warped.astype(np.uint8)
        return out


    def __call__(self, seg_map):
        device = seg_map.device

        if type(seg_map) == torch.Tensor:
            seg_map = seg_map.cpu().detach().numpy()

        seg_map_copy = seg_map.copy()

        dst_comp = self.random_choice(seg_map_copy, smallest_top=self.smallest_top)
        dst_mask = self.get_component_mask(seg_map_copy, dst_comp)
        ground_id = self.get_neighborhood_id(dst_mask, seg_map_copy)
        src_comp = random.choice([i for i in range(self.num_cls) if i not in [dst_comp, ground_id]])

        if random.choice(['Aug', 'None']) == 'Aug':
            aug_dst_mask = self.mask_aug(dst_mask)
            seg_map_copy[dst_mask.astype(bool)] = ground_id
            seg_map_copy[aug_dst_mask.astype(bool)] = src_comp
            mask = np.zeros_like(seg_map_copy).astype(float)
            mask[seg_map_copy == src_comp] = 0.5
            mask[seg_map_copy != seg_map] = 1
        else:
            seg_map_copy[dst_mask.astype(bool)] = src_comp
            mask = np.zeros_like(seg_map_copy).astype(float)
            mask[seg_map_copy == src_comp] = 0.5
            mask[seg_map_copy != seg_map] = 1

        return torch.from_numpy(seg_map_copy).to(device), torch.from_numpy(mask).to(device)



class ComponentCutPaste(BaseSynthesizer):

    def __init__(self, add_percent, smallest_top=None, **kwargs):
        super().__init__()
        self.add_percent = add_percent
        self.smallest_top = smallest_top

    def cutpaste_by_mask(self, src_mask, seg_map, threshold = 0.9):

        H, W = src_mask.shape

        rows, cols = np.where(src_mask == 1)
        top, left = rows.min(), cols.min()
        bottom, right = rows.max(), cols.max()
        height = bottom - top + 1
        width = right - left + 1
        if height >= int(H * threshold) or width >= int(W * threshold):
            return seg_map

        dst_left = random.randint(0, W - width)
        dst_top = random.randint(0, H - height)
        dst_mask = np.zeros_like(src_mask)
        dst_mask[dst_top: dst_top + height, dst_left: dst_left + width] = src_mask[top: top + height, left: left + width]
        seg_map[dst_mask.astype(bool)] = seg_map[src_mask.astype(bool)]
        return seg_map

    def __call__(self, seg_map):
        device = seg_map.device

        if type(seg_map) == torch.Tensor:
            seg_map = seg_map.cpu().detach().numpy()

        seg_map_copy = seg_map.copy()
        comp_id = self.random_choice(seg_map_copy, smallest_top= self.smallest_top)
        comp_mask = self.get_component_mask(seg_map_copy, comp_id)
        areas = self.get_connected_areas(comp_mask)

        if len(areas) > 1:
            erase_areas = random.sample(areas, max(1, int(len(areas) * self.add_percent)))
            for area in erase_areas:
                seg_map_copy = self.cutpaste_by_mask(area, seg_map_copy)
        else:
            seg_map_copy = self.cutpaste_by_mask(areas[0], seg_map_copy)

        mask = np.zeros_like(seg_map_copy).astype(float)
        if np.any(seg_map_copy != seg_map):
            mask[seg_map_copy == comp_id] = 1

        return torch.from_numpy(seg_map_copy).to(device), torch.from_numpy(mask).to(device)



SYNTHESIZER_MAP = {
    'DeleteComponents': DeleteComponents,
    'ReplaceComponents': ReplaceComponents,
    'ComponentCutPaste': ComponentCutPaste,
    'EraseComponents': EraseComponents,
}

class CombSynthesizer:

    def __init__(self, configs, num_cls, ano_percent = 0.5):

        self.synthesizers = []
        for name in configs:
            if name in ['DeleteComponents', 'EraseComponents', 'ComponentCutPaste']:
                if configs[name] is not None:
                    self.synthesizers.append(SYNTHESIZER_MAP[name](**configs[name]))
                else:
                    self.synthesizers.append(SYNTHESIZER_MAP[name]())
            elif name in ['ReplaceComponents'] and num_cls > 2:
                if configs[name] is not None:
                    self.synthesizers.append(SYNTHESIZER_MAP[name](num_cls = num_cls, **configs[name]))
                else:
                    self.synthesizers.append(SYNTHESIZER_MAP[name](num_cls = num_cls))

        assert ano_percent > 0 and ano_percent < 1
        self.ano_percent = ano_percent
        self.num_cls = num_cls

    @torch.no_grad()
    def __call__(self, seg_map):
        if self.num_cls == 1:
            return seg_map.clone(), torch.zeros_like(seg_map)
        else:
            if seg_map.ndim == 2:
                synthesizer = random.choice(self.synthesizers)
                return synthesizer(seg_map)
            elif seg_map.ndim == 3:
                B = seg_map.shape[0]
                seg_map_aug = []
                masks = []
                ano_num = int(B * self.ano_percent)

                for i in range(ano_num):
                    cluster_id_aug, mask = random.choice(self.synthesizers)(seg_map[i])
                    seg_map_aug.append(cluster_id_aug)
                    masks.append(mask)

                seg_map_aug = torch.cat([torch.stack(seg_map_aug), seg_map[ano_num: ]], dim=0)
                masks = torch.cat([torch.stack(masks), torch.zeros((B - ano_num, seg_map.shape[1], seg_map.shape[2])).to(seg_map.device)])
                return seg_map_aug, masks
