1. Create point and raster feature
    python feature_point.py --gpu 1 --model_path ../1_xyz/log/12301030_ds4020_weight_1000/best_model.ckpt-179 --output_path /result_output/3_feature_level/data/ds4020
	python feature_raster.py
	
2. Create dataset for train
    python create_train_dataset.py

3. train 2_layer fusionnet work
    python fusionnet_train.py
	
4. infence result
    python fusionnet_inference.py
	
