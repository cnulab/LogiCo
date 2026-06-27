### Download Datasets  
- MVTec-LOCO [[Official]](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad)
- MVTec-AD [[Official]](https://www.mvtec.com/company/research/datasets/mvtec-ad/)  
- VisA [[Official]](https://github.com/amazon-science/spot-diff)
- Real-IAD [[Official]](https://realiad4ad.github.io/Real-IAD/)
  
Place them in this folder. MVTec-AD and MVTec-LOCO do not require preprocessing. For VisA, please use the following command for preprocessing:
```
$ python prepare_visa.py --data_folder <Downloaded-VisA-Dataset> --save_folder visa  --split_file split_csv/1cls.csv
```
For Real-IAD, use the following command:
```
$ python prepare_realiad.py --data_folder <Downloaded-Real-IAD-Dataset> --save_folder realiad  --split_file_root realiad_jsons_sv
```

### Download Segmentation Maps 
- [mvtec_loco segmentations](https://drive.google.com/file/d/1oUcC3O_2-T6TccEdCEC4VBk9mChIffAm/view?usp=drive_link)
- [mvtec segmentations](https://drive.google.com/file/d/183KJr-mFXqaZkIo6grc6ZG47loa_XPqt/view?usp=drive_link)  
- [visa segmentations](https://drive.google.com/file/d/1w8OLhOdh5vt39nlwuBxdkM4W4Rxl1z4Y/view?usp=drive_link)
- [real-iad segmentations](https://drive.google.com/file/d/1oIpjoi9jKJvlZ_LeIKSg6qp_UmYhUsa-/view?usp=drive_link)

### Directory Structure 
  ```
    |--data                       
        |--mvtec
            |--bottle
              |--ground_truth
              |--test
              |--train
            |--cable
            |--...
        |--visa
            |--candle
              |--ground_truth
              |--test
              |--train
            |--capsules
            |--...
        |--mvtec_loco
            |--breakfast_box
              |--ground_truth
              |--test
              |--train
              |--validation
              |--defects_config.json
            |--juice_bottle
            |--...
        |--realiad
            |--audiojack
              |--ground_truth
              |--test
              |--train
            |--bottle_cap
            |--...
        |--mvtec_segmentations
            |--bottle
              |--test
              |--train
            |--cable
            |--...
        |--visa_segmentations
        |--mvtec_loco_segmentations
        |--realiad_segmentations
```
