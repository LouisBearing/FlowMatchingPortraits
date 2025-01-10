#! /bin/sh

python HMo_audio/lip_syncer_train.py \
	--data_dir='vox/train' \
	--keypoints \
	--audio_style=1 \
	--embed_style_x='mlp' \
	--e_x='conv1d_x' \
	--e_a='conv1d_a' \
	--lr_gamma=1 \
	--lr=1e-5 \
	--loss='infonce' \
	--delta=1e-3 \
	--pyramid_level=3 \
	--pyramid_kernel_size=7 \
	--batch_size=16 \
	--num_workers=4 \
	--steps=200000 \
	--out_dir='KP_vox_lvl3'