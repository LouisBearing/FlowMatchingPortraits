import numpy as np
import pandas as pd
import imageio
import os
import subprocess
from multiprocessing import Pool
from itertools import cycle
import warnings
import glob
import time
from tqdm import tqdm
from util import save, load_KPDect
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
import librosa
import soundfile as sf
import pickle
warnings.filterwarnings("ignore")

DEVNULL = open(os.devnull, 'wb')
conf_p = 'fomm_models/vox-adv-256.yaml'
chkpt_p = 'fomm_models/vox-adv-cpk.pth.tar'

def download(video_id, args):
    video_path = os.path.join(args.video_folder, video_id + "temp" + ".mp4")
    subprocess.call(["./yt-dlp", '-f', "''best/mp4''", '--write-auto-sub', '--write-sub',
                     '--sub-lang', 'en', '--skip-unavailable-fragments',
                     "https://www.youtube.com/watch?v=" + video_id, "--output",
                     video_path], stdout=DEVNULL, stderr=DEVNULL)
    return video_path
# https://www.youtube.com/watch?v=bd0jqkNxNdU
# bd0jqkNxNdU
def run(data):
    
    video_id, args = data
    t_bool = True # args.first_frame_for_test > 0

    # if args.keypoints:
    #     kp_detector = load_KPDect(conf_p, chkpt_p)
    #     suff = '_kp'
    # else:
    #     fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
    #     suff = '_ldk'
    # if args.flip:
    #     suff += 'flip'
    suff = '.png'

    # Checking if already processed
    processed_vids = []
    for partition in ['test', 'train']:
        processed_vids.extend(np.unique([f.split('#')[1] for f in glob.glob(os.path.join(args.out_folder, partition) + '/*.png')]))
    
    # if args.first_frame_for_test > 1:
    #     dirs = glob.glob1(args.out_folder, 'id*')
    #     processed_vids = np.unique([f.split('#')[1] for f in dirs])
    # else:
    #     walk = os.walk(args.out_folder)
    #     vid_id_list = []
    #     for _, _, files in walk:
    #         vid_id_list.extend([lef for lef in files if lef.endswith(suff)])
    #     processed_vids = np.unique([f.split('#')[1] for f in vid_id_list])
    if video_id.split('#')[0] in processed_vids:
        print(f'~~~~Vid id {video_id} already done, going to next vid~~~~')
        return
    else:
        print(f'---> Processing {video_id}...')

    # Audio
    if (not t_bool) and (not args.ignore_audio):
        librosa_data, freq = librosa.load(vid_path, sr=None)

    # ydl_opts = {
    #     'format': 'bestaudio/best',
    #     'postprocessors': [{
    #         'key': 'FFmpegExtractAudio',
    #         'preferredcodec': 'mp3',
    #         'preferredquality': '192',
    #     }],
    #     'progress_hooks': [my_hook],
    # }
    # vid_path = os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')
    # ydl = youtube_dl.YoutubeDL(ydl_opts)
    # with ydl:
    #     ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

    df = pd.read_csv(args.metadata)
    df = df[df['video_id'] == video_id]
    
    all_chunks_dict = [{'start': df['start'].iloc[j], 'end': df['end'].iloc[j],
                        'bbox': list(map(int, df['bbox'].iloc[j].split('-'))), 'frames':[], 'img':[],
                        'img_counter': 0} for j in range(df.shape[0])]
    ref_fps = df['fps'].iloc[0]
    ref_height = df['height'].iloc[0]
    ref_width = df['width'].iloc[0]
    partition = df['partition'].iloc[0]

    # if t_bool and partition == 'train':
    #     return

    vid_path = os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')
    print(f'~~downloading video {video_id} to path {vid_path}~~')
    download(video_id.split('#')[0], args)
    print(f'**Done downloading {video_id}**')

    temp_video_path = os.path.join(args.video_folder, video_id.split('#')[0] + "temp" + ".mp4")
    command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (temp_video_path,vid_path))
    output = subprocess.call(command, shell=True, stdout=None)
    # os.remove(temp_video_path)

    if not os.path.exists(vid_path):
       print ('Can not load video %s, broken link' % video_id.split('#')[0])
       return

    reader = imageio.get_reader(vid_path)
    fps = reader.get_meta_data()['fps']

    for i, frame in enumerate(reader):
        for entry in all_chunks_dict:
            # cond1 = t_bool and ((i * ref_fps >= entry['start'] * fps) and ((i - 1) * ref_fps < entry['start'] * fps))
            cond1 = t_bool and ((i * ref_fps >= entry['start'] * fps) and (i * ref_fps < entry['end'] * fps)) \
                and (entry['img_counter'] < 1) # args.first_frame_for_test
            cond2 = (not t_bool) and ((i * ref_fps >= entry['start'] * fps) and (i * ref_fps < entry['end'] * fps))
            if cond1 or cond2:
                left, top, right, bot = entry['bbox']
                left = int(left / (ref_width / frame.shape[1]))
                top = int(top / (ref_height / frame.shape[0]))
                right = int(right / (ref_width / frame.shape[1]))
                bot = int(bot / (ref_height / frame.shape[0]))
                crop = frame[top:bot, left:right]
                if args.image_shape is not None:
                    crop = img_as_ubyte(resize(crop, args.image_shape, anti_aliasing=True))

                # if args.flip:
                #     crop = np.flip(crop, axis=1)
                
                # if args.keypoints:
                #     with torch.no_grad():
                #         torch_crop = torch.tensor(crop[np.newaxis].astype(np.float32) / 255).permute(0, 3, 1, 2).cuda()
                #         kp = kp_detector(torch_crop)
                #     if kp['value'] is None:
                #         print('Couldnt detect keypoints')
                #         entry['frames'].append(np.zeros((10, 6)))
                #     else:
                #         entry['frames'].append((torch.cat([kp['value'], kp['jacobian'].flatten(start_dim=-2)], dim=-1)[0].cpu().numpy()))
                # else:
                #     landmarks = fa.get_landmarks(crop)
                #     if landmarks is None:
                #         print('Couldnt detect landmarks')
                #         entry['frames'].append(np.zeros((68, 3)))
                #     else:
                #         entry['frames'].append(landmarks[0])
                if t_bool:
                    entry['img'].append(crop)
                    entry['img_counter'] += 1
    
    for entry in all_chunks_dict:
        if (not t_bool) and (not args.ignore_audio):
            s_idx = int(freq * entry['start'] / ref_fps)
            e_idx = int(freq * entry['end'] / ref_fps)
            entry['audio'] = librosa_data[s_idx:e_idx]

        if 'person_id' in df:
            first_part = df['person_id'].iloc[0] + "#"
        else:
            first_part = ""
        first_part = first_part + '#'.join(video_id.split('#')[::-1])
        # Save landmarks
        try:
            path = first_part + '#' + str(entry['start']).zfill(6) + '#' + str(entry['end']).zfill(6)
            if t_bool:
                # if args.first_frame_for_test > 1:
                #     save_dir = os.path.join(args.out_folder, path.replace('#', '___'))
                #     os.makedirs(save_dir, exist_ok=True)
                #     for img_idx, frame in enumerate(entry['img']):
                #         imageio.imwrite(os.path.join(save_dir, format(img_idx, '05d')  + ".png"), frame)
                # else:
                #     with open(os.path.join(args.out_folder, path + '_ldk0'), 'wb') as f:
                #         pickle.dump(entry['frames'][0], f)
                if len(entry['img']) > 0:
                    imageio.imwrite(os.path.join(args.out_folder, partition, path + '.png'), entry['img'][0]) 
            else:
                with open(os.path.join(args.out_folder, partition, path + suff), 'wb') as f:
                    pickle.dump(np.stack(entry['frames']), f)
                if not args.ignore_audio:
                    # Save audio
                    sf.write(os.path.join(args.out_folder, partition, path + '.wav'), entry['audio'], freq)
        except ValueError:
            continue

    rem = [os.remove(os.path.join(args.video_folder, f)) for f in os.listdir(args.video_folder) if video_id.split('#')[0] in f]

    print(f'~~ {video_id} processing over <---')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video_folder", default='youtube-taichi', help='Path to youtube videos')
    parser.add_argument("--metadata", default='taichi-metadata-new.csv', help='Path to metadata')
    parser.add_argument("--out_folder", default='taichi-png', help='Path to output')
    parser.add_argument("--format", default='.png', help='Storing format')
    parser.add_argument("--workers", default=1, type=int, help='Number of workers')
    parser.add_argument("--youtube", default='./youtube-dl', help='Path to youtube-dl')
    parser.add_argument("--first_frame_for_test", default=0, type=int, help='Last preprocessing step for test data')
    parser.add_argument("--keypoints", default=False, action='store_true', help='KP or landmarks')
    parser.add_argument("--flip", default=False, action='store_true', help='Horizontal flip for data augmentation')
    parser.add_argument("--ignore_audio", default=False, action='store_true', help='Don\'t save audio')
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape, None for no resize")

    args = parser.parse_args()

    # torch.multiprocessing.set_start_method('spawn')

    if not os.path.exists(args.video_folder):
        os.makedirs(args.video_folder)
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    for partition in ['test', 'train']:
        if not os.path.exists(os.path.join(args.out_folder, partition)):
            os.makedirs(os.path.join(args.out_folder, partition))

    df = pd.read_csv(args.metadata)
    video_ids = set(df['video_id'])
    pool = Pool(processes=args.workers)
    args_list = cycle([args])
    for chunks_data in tqdm(pool.imap_unordered(run, zip(video_ids, args_list))):
        None  
