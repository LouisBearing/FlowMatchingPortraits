#! /bin/sh

python av_syncer_train.py \
	--data_dir='vox_lia_features_ds/train' \
	--audio_dir='vox_audio_files/train' \
	--audio_style=1 \
	--e_x='conv1d_x' \
	--e_a='conv1d_a' \
	--lr_gamma=1 \
	--lr=1e-5 \
	--loss='infonce' \
	--delta=1e-3 \
	--pyramid_level=0 \
	--pyramid_kernel_size=7 \
	--batch_size=16 \
	--num_workers=8 \
	--steps=1900000 \
	--out_dir='av_syncer_lvl0'
