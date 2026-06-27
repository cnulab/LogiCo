python create_segmentation_maps.py --dataset mvtec_loco --category breakfast_box
python create_segmentation_maps.py --dataset mvtec_loco --category juice_bottle
python create_segmentation_maps.py --dataset mvtec_loco --category pushpins
python create_segmentation_maps.py --dataset mvtec_loco --category screw_bag
python create_segmentation_maps.py --dataset mvtec_loco --category splicing_connectors

python train_logico.py --dataset mvtec_loco --category breakfast_box
python train_logico.py --dataset mvtec_loco --category juice_bottle
python train_logico.py --dataset mvtec_loco --category pushpins
python train_logico.py --dataset mvtec_loco --category screw_bag
python train_logico.py --dataset mvtec_loco --category splicing_connectors

python evaluation.py --dataset mvtec_loco --category breakfast_box
python evaluation.py --dataset mvtec_loco --category juice_bottle
python evaluation.py --dataset mvtec_loco --category pushpins
python evaluation.py --dataset mvtec_loco --category screw_bag
python evaluation.py --dataset mvtec_loco --category splicing_connectors