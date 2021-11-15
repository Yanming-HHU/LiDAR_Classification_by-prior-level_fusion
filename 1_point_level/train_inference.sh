#！/bin/bash
# python train.py --data_dir train_data/ --log_dir log/ -max_epoch 500
python inference.py --model_path log/best_model.ckpt-37600 --input_path data/ --output_type BOTH --output_path inference/
