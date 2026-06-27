"""Anomaly metrics."""
import os
import numpy as np
import json
from sklearn import metrics
from numpy import ndarray
import pandas as pd
from skimage import measure
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils.spro.image import DefectsConfig
from utils.spro.spro import read_maps, get_auc_spro_results
from utils.spro.aggregation import MetricsAggregator


def compute_imagewise_log_metrics(
    pr_image_list, gt_image_list, img_paths, **kwargs
):

    y_true = {"good": [], "logical_anomalies": [], "structural_anomalies": []}
    y_score = {"good": [], "logical_anomalies": [], "structural_anomalies": []}

    for gt_score, pred_score, img_path in zip(gt_image_list, pr_image_list, img_paths):
        if img_path.find('good') != -1:
            y_true['good'].append(gt_score)
            y_score['good'].append(pred_score)
        elif img_path.find('logical_anomalies') != -1:
            y_true['logical_anomalies'].append(gt_score)
            y_score['logical_anomalies'].append(pred_score)
        else:
            assert img_path.find('structural_anomalies') != -1
            y_true['structural_anomalies'].append(gt_score)
            y_score['structural_anomalies'].append(pred_score)

    auroc_str = roc_auc_score(y_true['good'] + y_true['structural_anomalies'],
                              y_score['good'] + y_score['structural_anomalies'])
    auroc_log = roc_auc_score(y_true['good'] + y_true['logical_anomalies'],
                              y_score['good'] + y_score['logical_anomalies'])
    auroc_all = (auroc_str + auroc_log) / 2.0

    return {"Log-iAUC": auroc_log, "Str-iAUC": auroc_str, "mean-iAUC": auroc_all}


def compute_imagewise_metrics(
    pr_image_list, gt_image_list, **kwargs
):
    """
    Computes image-wise metrics (Image Auroc).

    Args:
        prediction_scores: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        gt_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    auroc = metrics.roc_auc_score(
        gt_image_list, pr_image_list
    )

    return {"iAUC": auroc}


def compute_pixelwise_metrics(pr_pixel_list, gt_pixel_list, metrics_name = None, **kwargs):
    """
    Computes pixel-wise metrics (Pixel-AUROC, AP, F1) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        prediction_masks: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        gt_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(pr_pixel_list, list):
        pr_pixel_list = np.stack(pr_pixel_list)
    if isinstance(gt_pixel_list, list):
        gt_pixel_list = np.stack(gt_pixel_list)

    flat_anomaly_segmentations = pr_pixel_list.ravel()
    flat_ground_truth_masks = gt_pixel_list.ravel()

    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    ap = average_precision_score(flat_ground_truth_masks, flat_anomaly_segmentations)
    precision, recall, thresholds = precision_recall_curve(flat_ground_truth_masks, flat_anomaly_segmentations)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    f1 = f1[np.argmax(f1)]
    results = {
        "pAUC": auroc,
        "pAP": ap,
        "pF1" : f1,
    }
    if metrics_name is None:
        return results
    else:
        return {key: results[key] for key in results if key in metrics_name}


def compute_pro(pr_pixel_list, gt_pixel_list, num_th: int = 200, **kwargs):
    """
    Computes pixel-wise metrics (Pixel-AUROC, AP, F1) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        prediction_masks: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        gt_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
     Args:
         prediction_masks (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
         gt_masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
         num_th (int, optional): Number of thresholds
     """
    assert isinstance(pr_pixel_list, ndarray), "type(amaps) must be ndarray"
    assert isinstance(gt_pixel_list, ndarray), "type(masks) must be ndarray"
    assert pr_pixel_list.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert gt_pixel_list.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert pr_pixel_list.shape == gt_pixel_list.shape, "amaps.shape and masks.shape must be same"
    assert set(gt_pixel_list.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(pr_pixel_list, dtype=bool)
    min_th = pr_pixel_list.min()
    max_th = pr_pixel_list.max()
    delta = (max_th - min_th) / num_th
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[pr_pixel_list <= th] = 0
        binary_amaps[pr_pixel_list > th] = 1
        pros = []
        for binary_amap, mask in zip(binary_amaps,  gt_pixel_list):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)
        inverse_masks = 1 - gt_pixel_list
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        df_new = pd.DataFrame([{"pro": np.mean(pros), "fpr": fpr, "threshold": th}])
        df = pd.concat([df, df_new], ignore_index=True)
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()
    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return {"PRO": pro_auc}


def compute_pixelwise_sPRO_metrics(
    pr_pixel_list, img_paths, data_root, **kwargs
):

    defects_config_path = os.path.join(data_root, 'defects_config.json')
    with open(defects_config_path) as defects_config_file:
        defects_list = json.load(defects_config_file)
    defects_config = DefectsConfig.create_from_list(defects_list)

    gt_maps, anomaly_maps = read_maps(
        img_paths = img_paths,
        anomaly_maps = pr_pixel_list,
        defects_config = defects_config,
        category = os.path.split(data_root)[-1],
    )

    metrics_aggregator = MetricsAggregator(
            gt_maps=gt_maps,
            anomaly_maps=anomaly_maps,
            parallel_workers = None,
            parallel_niceness = 20)

    metrics = metrics_aggregator.run(
        curve_max_distance = 0.001, max_steps=10)

    localization_results = get_auc_spro_results(metrics=metrics)

    return {"Log-sPRO": (localization_results['auc_spro']['logical_anomalies'][0.05]).item(),
            "Str-sPRO": (localization_results['auc_spro']['structural_anomalies'][0.05]).item(),
            "mean-sPRO": (localization_results['auc_spro']['mean'][0.05]).item()}


def metrics2str(results):
    return ", ".join(["{}: {:.2f}".format(key, results[key] * 100.0) for key in results])
