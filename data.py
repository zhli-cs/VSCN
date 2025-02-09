"""Data provider"""
#!/usr/bin/env python

import json
import math
import os
import os.path as osp
import pickle
import random
import sys
from copy import deepcopy
from glob import glob
from time import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from nltk.tokenize import word_tokenize
from PIL import Image
from torch.utils.data import Dataset


class region_loader(Dataset):
    """
    Load precomputed captions and image features
    """
    def __init__(self, imdb): 
        self.opt = imdb.cfg
        self.db = imdb  
        
    def __len__(self):
        return len(self.db.scenedb)

    def __getitem__(self, scene_index):
        scene = self.db.scenedb[scene_index]  
        image_index = scene['image_index']  

        # Load region feature
        region_path = self.db.region_path_from_index(image_index)  
        with open(region_path, 'rb') as fid:
            regions = pickle.load(fid, encoding='latin1') 
        region_boxes = torch.from_numpy(regions['region_boxes']).float()  
        region_feats = torch.from_numpy(regions['region_feats']).float()  
        
        # Load CLIP-encoded image feature
        global_path = self.db.global_path_from_index(image_index)
        global_feats = np.load(global_path, encoding='latin1')  
        global_feats = torch.from_numpy(global_feats).float()

        # Load region caption
        all_meta_regions = [scene['regions'][x] for x in sorted(list(scene['regions'].keys()))]  
        all_captions = [x['caption'] for x in all_meta_regions]  

        if self.db.split in ['val', 'test']:
            captions = all_captions[:self.opt.max_turns]  
        else:
            num_captions = len(all_captions)
            caption_inds = np.random.permutation(range(num_captions))  
            captions = [all_captions[x] for x in caption_inds[:self.opt.max_turns]]  

        # Load CLIP-encoded query feature
        caption_feature_path = osp.abspath(osp.join(self.opt.data_path,'vg/caption_embedding_clip_vit_base16',self.db.split,'%d.npy'%scene_index))  
        all_caption_feature=np.load(caption_feature_path, encoding='latin1')   
        all_caption_feature = torch.from_numpy(all_caption_feature).float() 
        if self.db.split in ['val', 'test']:
            caption_feature = all_caption_feature[:self.opt.max_turns]  
        else:
            caption_feature_inds = caption_inds  
            caption_feature = [all_caption_feature[x] for x in caption_feature_inds[:self.opt.max_turns]]  

        return region_feats, caption_feature, global_feats, scene_index


def region_collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, caption, global_features, img_id) tuples.
    """
    region_feats, caption_feature, global_feats, scene_indices = zip(*data)
    bsize=len(region_feats)
    # regions
    max_region_num = len(region_feats[0])  
    new_region_feats = torch.zeros(len(region_feats), max_region_num, region_feats[0].size(-1)).float() 
    new_global_feats = torch.zeros(len(global_feats), global_feats[0].size(-1)).float()  
    new_caption_feature  = torch.zeros(len(caption_feature), len(caption_feature[0]), caption_feature[0][0].size(-1)).float()  
    temp=new_caption_feature  

    for i in range(len(region_feats)): 
        end = region_feats[i].size(0)  
        new_region_feats[i, :end] = region_feats[i]  

    for i in range(len(global_feats)): 
        end = global_feats[i].size(0)  
        new_global_feats[i] = global_feats[i]  

    for i in range(len(caption_feature)):  
        caption_embedding_end = len(caption_feature[i][0])  
        for j in range(len(caption_feature[0])):  
            temp[i, j, :caption_embedding_end] = caption_feature[i][j] 
        caption_nturns_end = len(caption_feature[i])  
        new_caption_feature[i, :caption_nturns_end] = temp[i] 

    return new_region_feats, new_caption_feature, new_global_feats, scene_indices

def get_precomp_loader(db, batch_size=100, shuffle=True, num_workers=4, pin_memory=False):

    dset = region_loader(db)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=pin_memory,
                                              num_workers=num_workers,
                                              collate_fn=region_collate_fn)
    return data_loader


def get_loaders(train_db, val_db, batch_size, workers, pin_memory):
 
    # get the train_loader
    train_loader = get_precomp_loader(train_db, batch_size, True, workers, pin_memory)
    # get the val_loader
    val_loader = get_precomp_loader(val_db, 100, False, workers, pin_memory)

    return train_loader, val_loader


def get_test_loader(test_db, workers, pin_memory):

    # get the val/test_loader
    test_loader = get_precomp_loader(test_db, 100, False, workers, pin_memory)

    return test_loader
