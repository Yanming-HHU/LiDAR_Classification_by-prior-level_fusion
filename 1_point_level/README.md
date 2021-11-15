1 create train data<br>:
	python create_train_dataset.py --output-path ./data
	
2 train pointnet++ using<br>
	python train.py --data_dir ./data --log_dir ./log/0913_raw_training --max_epoch 501
	tensorboard --logdir=./log/0913_raw_training

3 Inference<br>
	python inference_isprs.py --input_path ../data --output_path ./inference --model_path ./log/0912/best_model
