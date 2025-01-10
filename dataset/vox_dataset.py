import torch, os, glob, pickle
import numpy as np
from torch.utils.data import Dataset
# from .utils import *

######
####
### La classe principale pour le chargement des données
####
# TODO: Lightning ?
# Garder la possiblité de tester sur keypoints de FOMM (en sécurité, dans un 1er temps)
#######



class VoxDataset(Dataset):
    
    def __init__(self, motion_feat_dir, audio_dir, test=False, pyramid_level=0, kernel_size=2):
        super(VoxDataset, self).__init__()

        ### Instantiation paramètres de classe

        self.motion_feat_dir = motion_feat_dir
        self.audio_dir = audio_dir

        self.transform = not test

        # Ces 2 derniers paramètres permettent de construire la pyramide gaussienne directement dans le dataloader
        self.pyramid_level = pyramid_level
        self.kernel_size = kernel_size

        # --------

        ## Problème avec le sample suivant pour les landmarks
        # -> TODO prévoir un méchanisme pour écarter les samples défaillants
        # forbidden_id = 'id10292#ENIHEvg_VLM' ## For some reason there's an issue with this one

        # --------

        ## Liste des ids qui forment le dataset
        self.vid_id = glob.glob(motion_feat_dir + '/*lia_feat')


    def __len__(self):
        return len(self.vid_id)
    
    
    def __getitem__(self, idx):

        ###
        ## Augmentation par symmétrie
        ###

        # Maybe flip
        flip = False
        if self.transform:
            if np.random.randint(low=0, high=2) == 1:
                flip = True

        # --------

        ###
        ## Loading du sample à partir de l'id 
        ###

        ### Load LIA features
        motion_feat_path = self.vid_id[idx]
        if flip:
            motion_feat_path += '_flip'
        if not os.path.isfile(motion_feat_path):
            motion_feat_path = motion_feat_path.replace('_flip', '')
        with open(motion_feat_path, 'rb') as f:
            motion_feat = pickle.load(f).squeeze() # Shape len, 1, 20 --> len, 20

        # --------

        # ###
        # ### Smoothing (possible jitter après l'encoder qu'on veut supprimer), on garde ? 
        # ###   
        # motion_feat = moving_avg_with_reflect_pad(motion_feat, 3)
        
        # --------

        ###
        ## Data augmentation: TODO: à supprimer sauf si on expérimente avec les keypoints de FOMM
        ###
        # if self.transform:

        #     ## Rescaling
        #     motion_feat = rescale_kp(motion_feat)
        #     ## Translation
        #     motion_feat = translate_kp(motion_feat)

        # --------

        ##
        # Loading du fichier audio correspondant
        ##
        file_id = os.path.basename(motion_feat_path.replace('_flip', '')).replace('lia_feat', '')
        with open(os.path.join(self.audio_dir, file_id + '_audiofeats'), 'rb') as f:
            audio_feat = pickle.load(f)[:, -26:]

        # --------

        ###
        ## Data augmentation pour audio, on garde ?
        ###
        # if self.transform:
        #     # Random signal power shift
        #     rdm_inc_range = 0.1 * (audio.max() - audio.min()).item()
        #     rdm_inc = np.random.uniform(-rdm_inc_range, rdm_inc_range)
        #     audio += rdm_inc

        # --------

        ###
        ## Gaussian pyramid
        ###
        ## Time pyramid level selection
        if self.pyramid_level > 0:
            motion_feat = motion_feat.numpy()
        for _ in range(self.pyramid_level):
            motion_feat = moving_avg_with_reflect_pad(motion_feat, n=self.kernel_size)[::2]
            audio_feat = moving_avg_with_reflect_pad(audio_feat, n=self.kernel_size)[::2]

        # --------
        
        # Convert to tensor
        if self.pyramid_level > 0:
            motion_feat = torch.from_numpy(motion_feat)
        audio_feat = torch.from_numpy(audio_feat)

        return (motion_feat, audio_feat)


def moving_avg_with_reflect_pad(a, n):
    '''
    Moving average on axis 0
    '''
    if n == 0:
        return a
    n_pads = int((n - 1) / 2)
    padding = (n_pads, n_pads + 1),
    for _ in range(len(a.shape) - 1):
        padding += (0,0),
    b = np.pad(a, padding, mode='reflect')
    b = np.cumsum(b, axis=0)
    b[n:] = b[n:] - b[:-n]
    return b[n - 1:n - 1 + len(a)] / n



### TODO dessous: fonctions non utilisées à supprimer


def get_theta_y(ldks):
    # Draw final y orientation randomly in a fixed interval, then compute the required rotation around y-axis
    vect = (ldks[:, 16] - ldks[:, 0]).mean(axis=0)
    z_i = vect[2]
    r = np.sqrt(vect[2] ** 2 + vect[0] ** 2)
    theta_i = np.arcsin(-z_i / r)
    theta_y = np.random.uniform(low=-(np.pi / 3), high=np.pi / 3) - theta_i
    return theta_y
    

def get_theta_x(ldks):
    # Draw final x orientation randomly in a fixed interval, then compute the required rotation around x-axis
    vect = (ldks[:, 8] - ldks[:, 0]).mean(axis=0)
    z_i = vect[2]
    r = np.sqrt(vect[2] ** 2 + vect[1] ** 2)
    theta_i = np.arcsin(-z_i / r)
    theta_x = np.random.uniform(low=-np.pi / 3, high=-np.pi / 7) - theta_i
    return theta_x


def rescale(ldks, rescaling):
    length, chan, dim = ldks.shape
    ## Compute parameters of transformation matrix
    origin = 0.5 * (1 - rescaling) * np.ones(2)
    M = affine_matrix(rescaling, origin, dim)
    ## Rescale and center
    ldks = ldks.reshape(length * chan, dim)
    ldks = scale_and_translate(ldks, M).reshape(length, chan, dim)
    return ldks


def translate(ldks):
    length, chan, dim = ldks.shape
    ## Random translation
    o_x = np.random.uniform(low=-ldks[..., 0].min(), high=0.99 - ldks[..., 0].max())
    o_y = np.random.uniform(low=-ldks[..., 1].min(), high=0.99 - ldks[..., 1].max())
    M = affine_matrix(1, [o_x, o_y], dim)
    ldks = ldks.reshape(length * chan, dim)
    ldks = scale_and_translate(ldks, M).reshape(length, chan, dim)
    return ldks


def translate_kp(ldks, max_off=0.16):
    '''
    Input shape: len, nkps, 6 (2 kp coord + 4 jac params)
    max_off: by default, 8% of the input size
    '''
    # Translate kp
    off = np.round(np.random.uniform(-max_off, max_off, size=2), 4)
    tx, ty = int(off[0] * 128), int(off[1] * 128)
    off = np.array([off[0], off[1], 0., 0., 0., 0.])[np.newaxis, np.newaxis]
    # Translate src img
    tx_pos, tx_neg, ty_pos, ty_neg = max(tx, 0), -min(tx, 0), max(ty, 0), -min(ty, 0)
    return ldks + off


def rescale_kp(ldks):
    '''
    Both kp coord (zero-centerd) and jacobian matrix params are multiplied by the rescaling factor
    '''
    # Ldk rescaling
    rescaling = np.random.uniform(low=0.9, high=1.1)
    return ldks * rescaling