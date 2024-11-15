# A Contrastive Generalization Framework based on Semantic Augmentation for  Medical Image Segmentation


## Pytorch Implementation

### Download Dataset

#### Fundus
Download dataset [Fundus](https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view) (Provided by [DoFE](https://github.com/emma-sjwang/Dofe)) and put images in ```./dataset/fundus/```

#### Prostate
Download our pre-processed dataset [Prostate](https://drive.google.com/file/d/1sx2FpNySQNjU6_zBa4DPnb9RAmesN0P6/view?usp=sharing) (Originally Provided by [SAML](https://liuquande.github.io/SAML/)) and put data in ```./dataset/prostate/```


### Training and Testing

Train on Fundus Dataset (Target Domain 0)
```
cd code
python -W ignore train.py --data_root ../dataset --dataset fundus --domain_idxs 1,2,3 --test_domain_idx 0 --ram --rec --is_out_domain --consistency --consistency_type kd --save_path ../outdir/fundus/target0 --gpu 0
python -W ignore train.py --data_root ../dataset --dataset fundus --domain_idxs 0,2,3 --test_domain_idx 1 --ram --rec --is_out_domain --consistency --consistency_type kd --save_path ../outdir/fundus/target1 --gpu 0
python -W ignore train.py --data_root ../dataset --dataset fundus --domain_idxs 0,1,3 --test_domain_idx 2 --ram --rec --is_out_domain --consistency --consistency_type kd --save_path ../outdir/fundus/target2 --gpu 0
python -W ignore train.py --data_root ../dataset --dataset fundus --domain_idxs 0,1,2 --test_domain_idx 3 --ram --rec --is_out_domain --consistency --consistency_type kd --save_path ../outdir/fundus/target3 --gpu 0
```

## Acknowledgement
Our implementation is heavily drived from [RAM-DSIR](https://github.com/zzzqzhou/RAM-DSIR), [Fed-DG](https://github.com/liuquande/FedDG-ELCFS) and [DoFE](https://github.com/emma-sjwang/Dofe). Thanks to their great work.


