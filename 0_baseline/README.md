1 Create train and test dataset<br>
	python creat_dataset.py --output-path ./data
	
2 Train pointnet++<br>
	python train.py --data_dir ./isprs_data --log_dir ./log/1127 --max_epoch 501

3 Inference<br>
	python inference_isprs.py --model_path ./log/PATH --output_path ./inference/TIME_SPECIAL
