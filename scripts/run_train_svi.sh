python train.py --emb_size 512 --img_size 1024 --max_epoch 25 --data_root data --data_name CVOGL_SVI --beta 1.0 --savename model_svi --gpu 0,1 --batch_size 12 --num_workers 24 --print_freq 50 > logs/train_svi.log 2>&1 
