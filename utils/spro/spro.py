import os
import cv2
from typing import Optional, Iterable

from .aggregation import MetricsAggregator, ThresholdMetrics
from .image import GroundTruthMap, AnomalyMap, DefectsConfig
from .util import get_auc_for_max_fpr, listdir, set_niceness, compute_classification_auc_roc


IMAGE_SIZE = {
    'breakfast_box': (1600, 1280),
    'juice_bottle': (800, 1600),
    'pushpins': (1700, 1000),
    'screw_bag': (1600, 1100),
    'splicing_connectors': (1700, 850),
}  

MAX_FPRS = [0.05]

def read_maps(img_paths,
              anomaly_maps,
              category,
              defects_config: DefectsConfig):

    gt_maps, pred_maps = [], []
    for img_path, anomaly_map in zip(img_paths, anomaly_maps):

        img_root, img_name = os.path.split(img_path)
        _, sub_category = os.path.split(img_root)

        if sub_category in ['logical_anomalies', 'structural_anomalies']:

            gt_map_path = os.path.join(img_root.replace('test', 'ground_truth'), img_name[:img_name.rfind('.')])
            gt_map = GroundTruthMap.read_from_png_dir(
                png_dir=gt_map_path,
                defects_config=defects_config)
            gt_maps.append(gt_map)
        else:
            gt_maps.append(None)

        anomaly_map_resize = cv2.resize(anomaly_map, dsize=IMAGE_SIZE[category], interpolation=cv2.INTER_LINEAR)
        pred_map = AnomalyMap(anomaly_map_resize, img_path)
        pred_maps.append(pred_map)

    return gt_maps, pred_maps



def get_auc_spros_for_metrics(
        metrics: ThresholdMetrics,
        filter_defect_names_for_spro: Optional[Iterable[str]] = None):
    """Compute AUC sPRO values for a given ThresholdMetrics instance.

    Args:
        metrics: The ThresholdMetrics instance.
        filter_defect_names_for_spro: If not None, only the sPRO values from
            defect names in this sequence will be used. Does not affect the
            computation of FPRs!
    """
    auc_spros = {}
    for max_fpr in MAX_FPRS:
        try:
            fp_rates = metrics.get_fp_rates()
        except ZeroDivisionError:
            auc = None
        else:
            mean_spros = metrics.get_mean_spros(
                filter_defect_names=filter_defect_names_for_spro)
            auc = get_auc_for_max_fpr(fprs=fp_rates,
                                      y_values=mean_spros,
                                      max_fpr=max_fpr,
                                      scale_to_one=True)
        auc_spros[max_fpr] = auc
    return auc_spros



def get_auc_spros_per_subdir(metrics: ThresholdMetrics,
                             add_good_images):

    """Compute the AUC sPRO for images in subdirectories (usually "good",
    "structural_anomalies" and "logical_anomalies").

    If add_good_images is True, the images in the "good" subdirectory will be
    added to the images of each subdirectory for computing the corresponding
    AUC sPRO value. Hence, the result dict will not contain a "good" key.
    """
    aucs_per_subdir = {}

    good_images = [a for a in metrics.anomaly_maps if os.path.realpath(a.file_path).find('good') !=- 1]
    log_images = [a for a in metrics.anomaly_maps if os.path.realpath(a.file_path).find('logical_anomalies') != -1]
    str_images = [a for a in metrics.anomaly_maps if os.path.realpath(a.file_path).find('structural_anomalies') != -1]

    log_group = log_images + good_images
    str_group = str_images + good_images

    log_metrics = metrics.reduce_to_images(log_group)
    str_metrics = metrics.reduce_to_images(str_group)

    aucs_per_subdir['structural_anomalies'] = get_auc_spros_for_metrics(str_metrics)
    aucs_per_subdir['logical_anomalies'] = get_auc_spros_for_metrics(log_metrics)

    return aucs_per_subdir



def get_auc_spro_results(metrics: ThresholdMetrics):
    """Compute AUC sPRO values for all images, images in subdirectories and
    defect names.
    """
    # Compute the AUC sPRO for logical and structural anomalies.
    auc_spro = get_auc_spros_per_subdir(
        metrics=metrics,
        add_good_images=True)

    mean_spros = dict()
    for limit in auc_spro['structural_anomalies'].keys():
        auc_spro_structural = auc_spro['structural_anomalies'][limit]
        auc_spro_logical = auc_spro['logical_anomalies'][limit]
        mean = 0.5 * (auc_spro_structural + auc_spro_logical)
        mean_spros[limit] = mean
    auc_spro['mean'] = mean_spros

    return {'auc_spro': auc_spro}
