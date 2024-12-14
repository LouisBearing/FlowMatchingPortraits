import torch
import os
import numpy as np
import pickle
from torch.utils.data import Dataset
from .utils import *

######
####
### La classe principale pour le chargement des données
####
# TODO: A adapter pour le latent space de LIA / mettre au format lightning ?
# Garder la possiblité de tester sur keypoints de FOMM (en sécurité, dans un 1er temps)
#######



class VoxLipsDataset(Dataset): ## Renommer
    
    def __init__(self, dir_path, audio_dir=None, test=False, pyramid_level=0, kernel_size=2):
        super(VoxLipsDataset, self).__init__()

        ### Instantiatin paramètres de classe

        # self.dir_path = dir_path
        # self.audio_dir = dir_path
        # if audio_dir is not None:
        #     self.audio_dir = audio_dir

        # self.transform = not test

        ## Ces 2 derniers paramètres permettent de construire la pyramide gaussienne directement dans le dataloader
        # self.pyramid_level = pyramid_level
        # self.kernel_size = kernel_size

        # --------

        ## Problème avec le sample suivant pour les landmarks
        # -> TODO prévoir un méchanisme pour écarter les samples défaillants
        # forbidden_id = 'id10292#ENIHEvg_VLM' ## For some reason there's an issue with this one

        # --------

        ## Liste des ids qui forment le dataset
        # self.vid_id = [f.split(self.suff)[0] for f in os.listdir(dir_path) if f.endswith(self.suff) and not forbidden_id in f]


    def __len__(self):
        return len(self.vid_id)
    
    
    def __getitem__(self, idx):
        ###
        ## Augmentation par symmétrie: TODO: à supprimer, sauf si on garde la possibilité d'entrainer sur les
        ## keypoints de FOMM dans un 1er temps par sécurité 
        ###

        # # Maybe flip
        # if self.transform:
        #     if np.random.randint(low=0, high=2) == 1:
        #         self.suff = '_kpflipc'

        # --------

        ###
        ## Loading du sample à partir de l'id TODO: à remplacer 
        ###

        # ### Keypoints
        # fp = os.path.join(self.dir_path, self.vid_id[idx] + self.suff)
        # if not os.path.isfile(fp):
        #     fp = fp.replace('flip', '')
        # with open(fp, 'rb') as f:
        #     x = pickle.load(f)

        # --------

        ###
        ### Smoothing (possible jitter après l'encoder qu'on veut supprimer), on garde ? 
        ###   
        x = moving_avg_with_reflect_pad(x, 3)
        
        # --------

        ###
        ## Data augmentation: TODO: à supprimer sauf si on expérimente avec les keypoints de FOMM
        ###
        # if self.transform:

        #     ## Rescaling
        #     x = rescale_kp(x)
        #     ## Translation
        #     x = translate_kp(x)

        # --------

        ###
        ## TODO: Loading du fichier audio correspondant
        ###
        # with open(os.path.join(...), 'rb') as f:
        #     audio = pickle.load(f)

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
        ## Gaussian pyramid + to torch, on garde ?
        ###
        # ## Time pyramid level selection
        # for _ in range(self.pyramid_level):
        #     ldks = moving_avg_with_reflect_pad(ldks, n=self.kernel_size)[::2]
        #     audio = moving_avg_with_reflect_pad(audio, n=self.kernel_size)[::2]

        # --------
        
        # # Convert to tensor
        # sample = torch.Tensor(ldks)
        # melspec = torch.Tensor(audio)

        return (sample, melspec)
    

### TODO dessous: fonctions non utilisées à supprimer


def moving_avg_with_reflect_pad(a, n):
    '''
    Moving average on axis 0
    '''
    if n == 0:
        return a
    n_pads = int((n - 1) / 2)
    padding = (n_pads, n_pads + 1),
    for i in range(len(a.shape) - 1):
        padding += (0,0),
    b = np.pad(a, padding, mode='reflect')
    b = np.cumsum(b, axis=0)
    b[n:] = b[n:] - b[:-n]
    return b[n - 1:n - 1 + len(a)] / n


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