###
# Preprocessing file from FOMM
###
import numpy as np
import pandas as pd
import imageio
import os
import subprocess
from multiprocessing import Pool
from itertools import cycle
import warnings
import glob
from tqdm import tqdm
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
import pickle
import librosa
import soundfile as sf
warnings.filterwarnings("ignore")

from LIA.networks.generator import Generator
import torch

DEVNULL = open(os.devnull, 'wb')

def download(video_id, args):
    video_path = os.path.join(args.video_folder, video_id + "temp" + ".mp4")
    subprocess.call(["./yt-dlp", '--skip-unavailable-fragments',
                     "https://www.youtube.com/watch?v=" + video_id, "--output",
                     video_path], stdout=DEVNULL, stderr=DEVNULL)
    return video_path

def run(data):
    
    video_id, args = data

    suff = ''
    if args.flip:
        suff = '_flip'

    # Checking if already processed
    processed_vids = []
    search_key = '/*lia_feat' + suff
    if args.audio_only:
        search_key = '.wav'
    for partition in ['test', 'train']:
        processed_vids.extend(np.unique([f.split('#')[1] for f in glob.glob(os.path.join(args.out_folder, partition) + search_key)]))

    if video_id.split('#')[0] in processed_vids:
        print(f'~~~~Vid id {video_id} already done, going to next vid~~~~')
        return
    else:
        print(f'---> Processing {video_id}...')

    df = pd.read_csv(args.metadata)
    df = df[df['video_id'] == video_id]

    # Placeholder to avoid concurrent processes to work on the same video
    if 'person_id' in df:
        first_part = df['person_id'].iloc[0] + "#"
    else:
        first_part = ""
    first_part = first_part + '#'.join(video_id.split('#')[::-1])
    placeholder_path = os.path.join(args.out_folder, partition, first_part + '#' + '_placeholder_' + 'lia_feat' + suff)
    with open(placeholder_path, 'wb') as f:
        pickle.dump([], f)
    
    all_chunks_dict = [{'start': df['start'].iloc[j], 'end': df['end'].iloc[j],
                        'bbox': list(map(int, df['bbox'].iloc[j].split('-'))), 'lia_feat':[],
                        } for j in range(df.shape[0])]
    ref_fps = df['fps'].iloc[0]
    ref_height = df['height'].iloc[0]
    ref_width = df['width'].iloc[0]
    partition = df['partition'].iloc[0]

    vid_path = os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')
    print(f'~~downloading video {video_id} to path {vid_path}~~')
    download(video_id.split('#')[0], args)
    print(f'**Done downloading {video_id}**')

    temp_video_path = os.path.join(args.video_folder, video_id.split('#')[0] + "temp" + ".mp4")

    command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (temp_video_path, vid_path))
    output = subprocess.call(command, shell=True, stdout=None)
    # os.remove(temp_video_path)

    if not os.path.exists(vid_path):
       print ('Can not load video %s, broken link' % video_id.split('#')[0])
       return
    
    ####
    ### Either extract audio or LIA features
    ####

    ## Audio
    if args.audio_only:
        librosa_data, freq = librosa.load(vid_path, sr=None)
        for entry in all_chunks_dict:
            s_idx = int(freq * entry['start'] / ref_fps)
            e_idx = int(freq * entry['end'] / ref_fps)
            path = first_part + '#' + str(entry['start']).zfill(6) + '#' + str(entry['end']).zfill(6)
            sf.write(os.path.join(args.out_folder, partition, path + '.wav'), librosa_data[s_idx:e_idx], freq)

    ## LIA motion features
    else:
        # Load Lia model
        lia_gen = Generator(256, 512, 20, 1).cuda()
        weight = torch.load(args.model_path, map_location=lambda storage, loc: storage)['gen']
        lia_gen.load_state_dict(weight)
        lia_gen.eval()

        reader = imageio.get_reader(vid_path)
        fps = reader.get_meta_data()['fps']

        for i, frame in enumerate(reader):

            for entry in all_chunks_dict:

                if (i * ref_fps >= entry['start'] * fps) and (i * ref_fps < entry['end'] * fps):
                    left, top, right, bot = entry['bbox']
                    left = int(left / (ref_width / frame.shape[1]))
                    top = int(top / (ref_height / frame.shape[0]))
                    right = int(right / (ref_width / frame.shape[1]))
                    bot = int(bot / (ref_height / frame.shape[0]))
                    crop = frame[top:bot, left:right]

                    if args.image_shape is not None:
                        crop = img_as_ubyte(resize(crop, args.image_shape, anti_aliasing=True))

                    with torch.no_grad():
                        if args.flip:
                            flipped_crop = np.flip(crop, axis=1)
                            flipped_crop = torch.tensor(2 * flipped_crop.astype(np.float32) / 255 - 1.0).permute(2, 0, 1)[None].cuda()
                            lia_feats = lia_gen.enc.enc_motion(flipped_crop)
                        else:
                            crop = torch.tensor(2 * crop.astype(np.float32) / 255 - 1.0).permute(2, 0, 1)[None].cuda()
                            lia_feats = lia_gen.enc.enc_motion(crop)
                    entry['lia_feat'].append(lia_feats)
    
        for entry in all_chunks_dict:

            # Save features
            try:
                path = first_part + '#' + str(entry['start']).zfill(6) + '#' + str(entry['end']).zfill(6)
                with open(os.path.join(args.out_folder, partition, path + 'lia_feat' + suff), 'wb') as f:
                    pickle.dump(torch.stack(entry['lia_feat']), f)
            except:
                continue

    rem = [os.remove(os.path.join(args.video_folder, f)) for f in os.listdir(args.video_folder) if video_id.split('#')[0] in f]
    os.remove(placeholder_path)
    print(f'~~ {video_id} processing over <---')


if __name__ == "__main__":

    # video_id = 'cfG-zOBnY8Y'
    # subprocess.call(["./yt-dlp", '--write-auto-sub', '--write-sub',
    #                 '--sub-lang', 'en', '--skip-unavailable-fragments',
    #                 "https://www.youtube.com/watch?v=" + video_id, "--output",
    #                 'out.mp4'], stdout=DEVNULL, stderr=DEVNULL)

    # # subprocess.call(["yt-dlp",
    # #                 "https://www.youtube.com/watch?v=cfG-zOBnY8Y"], stdout=DEVNULL, stderr=DEVNULL)
    
    # # subprocess.call(["./yt-dlp",
    # #                 "https://www.youtube.com/watch?v=cfG-zOBnY8Y",
    # #                 "--output", "out.mp4"], stdout=DEVNULL, stderr=DEVNULL)

    parser = ArgumentParser()
    parser.add_argument("--video_folder", default='temp_folder_vox_dl', help='Path to youtube videos')
    parser.add_argument("--model_path", default='LIA/checkpoints/vox.pt', help='Path to Lia model')
    parser.add_argument("--metadata", default='vox-metadata.csv', help='Path to metadata')
    parser.add_argument("--out_folder", default='vox_lia_features_ds', help='Path to output')
    parser.add_argument("--workers", default=1, type=int, help='Number of workers')
    parser.add_argument("--flip", default=False, action='store_true', help='Horizontal flip for data augmentation')
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape, None for no resize")
    parser.add_argument('--audio_only', action='store_true', default=False)

    args = parser.parse_args()

    if not os.path.exists(args.video_folder):
        os.makedirs(args.video_folder)

    # print(f'DOWLOADING to {args.video_folder}')

    # vid = 'eFrsoyMcrc0'
    # video_path = os.path.join(args.video_folder, vid + "_temp" + ".mp4")
    # subprocess.call(["./yt-dlp",
    #                  "https://www.youtube.com/watch?v=" + vid, "--output",
    #                  video_path])
    # exists = os.path.exists(video_path)
    # print(f'Succes?: {exists}')

    df = pd.read_csv(args.metadata)
    video_ids = set(df['video_id'])
    pool = Pool(processes=args.workers)
    args_list = cycle([args])
    for chunks_data in tqdm(pool.imap_unordered(run, zip(video_ids, args_list))):
        None  
