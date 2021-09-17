# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for GPTrajectory.

References:
    https://github.com/karpathy/minGPT
"""
import numpy as np
import os
import math
import logging
import random
import datetime
import socket


import torch
import torch.nn as nn
from torch.nn import functional as F
torch.pi = torch.acos(torch.zeros(1)).item()*2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    
def new_log(logdir,filename):
    """Defines logging format.
    """
    filename = os.path.join(logdir,
                            datetime.datetime.now().strftime("log_%Y-%m-%d-%H-%M-%S_"+socket.gethostname()+"_"+filename+".log"))
    logging.basicConfig(level=logging.INFO,
                        filename=filename,
                        format="%(asctime)s - %(name)s - %(message)s",
                        filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)   
    
def haversine(input_coords, 
               pred_coords):
    """ Calculate the haversine distances between input_coords and pred_coords.
    
    Args:
        input_coords, pred_coords: Tensors of size (...,N), with (...,0) and (...,1) are
        the latitude and longitude in radians.
    
    Returns:
        The havesine distances between
    """
    R = 6371
    lat_errors = pred_coords[...,0] - input_coords[...,0]
    lon_errors = pred_coords[...,1] - input_coords[...,1]
    a = torch.sin(lat_errors/2)**2\
        +torch.cos(input_coords[:,:,0])*torch.cos(pred_coords[:,:,0])*torch.sin(lon_errors/2)**2
    c = 2*torch.atan2(torch.sqrt(a),torch.sqrt(1-a))
    d = R*c
    return d

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def top_k_nearest_idx(att_logits, att_idxs, r_vicinity):
    """Keep only k values nearest the current idx.
    
    Args:
        att_logits: a Tensor of shape (bachsize, data_size). 
        att_idxs: a Tensor of shape (bachsize, 1), indicates 
            the current idxs.
        r_vicinity: number of values to be kept.
    """
    device = att_logits.device
    idx_range = torch.arange(att_logits.shape[-1]).to(device).repeat(att_logits.shape[0],1)
    idx_dists = torch.abs(idx_range - att_idxs)
    out = att_logits.clone()
    out[idx_dists >= r_vicinity/2] = -float('Inf')
    return out