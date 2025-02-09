"""Evaluation"""

from __future__ import print_function

import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

import opts
from data import get_test_loader
from model import VSCN
from vg import vg
from vocab import Vocabulary

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # For values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # For stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # To keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # Create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    val_logger = LogCollector()

    # Switch to evaluate mode
    model.val_start()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    max_n_word = 0
    with torch.no_grad():
        for i, (image_feats, caption_feats, global_feats, ids) in enumerate(data_loader):
            # Make sure val logger is used
            model.logger = val_logger

            # Compute the embeddings
            img_emb, cap_emb, _, _ = model.forward_emb(image_feats, caption_feats)      
            bsize, max_turns, embed_size = cap_emb.size()
            if img_embs is None:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
                cap_embs = np.zeros((len(data_loader.dataset), max_turns, cap_emb.size(2)))
                global_all_feats = np.zeros((len(data_loader.dataset), global_feats.size(1)))

            # Cache embeddings
            ids = list(ids)
            img_embs[ids] = img_emb.data.cpu().numpy().copy()                                        
            cap_embs[ids] = cap_emb.data.cpu().numpy().copy()
            global_all_feats[ids] = global_feats.data.cpu().numpy().copy()
       
    return img_embs, cap_embs, global_all_feats                


def evalrank(model_path, split='test'):
    """
    Evaluate a trained model. 
    """
    # Load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    # opt = opts.parse_opt()
    save_epoch = checkpoint['epoch']
    print(opt)

    # Load dataset
    if 'val' in split:
        db = vg(opt, 'val')
    elif 'test' in split:
        db = vg(opt, 'test')

    # Build model
    model = VSCN(opt)

    # Load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(db, opt.workers, opt.pin_memory)
    print("=> loaded checkpoint_epoch {}".format(save_epoch))

    print('Computing results...')
    with torch.no_grad():  
        img_embs, cap_embs, global_feats = encode_data(model, data_loader)

        # Evaluate the retrieval performance of each round
        for i in range(opt.max_turns):
            print('Images: %d, Retrieval times: %d, %d queries for each retrieval' %
                (img_embs.shape[0]-96, cap_embs.shape[0]-96, i+1))
            
            # Determine the t_mask for current testing round
            indices = None
            N = img_embs.shape[0]  # the number of test samples
            t_mask_current_turn = torch.cat([torch.zeros(i+1), torch.ones(opt.max_turns-(i+1))], dim=0)  
            t_mask_current_turn_all = t_mask_current_turn.repeat(N, 1).cuda()  

            # Record computation time of validation
            start = time.time()
            sims = shard_attn_scores_test(model, img_embs, cap_embs, opt, indices, t_mask_current_turn_all, global_feats, current_test_turns=i+1, shared_size=200)
            end = time.time()
            print("calculate similarity time:", end-start)

            # Image retrieval
            ri, _ = eval_test(img_embs, cap_embs, sims, return_ranks=True)
            print("%d round " % (i+1), end='')
            print("image retrieval results: %.1f, %.1f, %.1f, %.1f" % ri)

def shard_attn_scores_val(model, img_embs, cap_embs, opt, indices, t_mask_all, global_feats, current_validate_turns=1, shared_size=200):
    n_im_shard = len(img_embs) // shared_size
    n_cap_shard = len(cap_embs) // shared_size

    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shared_size * i, min(shared_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
            ca_start, ca_end = shared_size * j, min(shared_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                t_mask = t_mask_all[ca_start:ca_end].to(torch.bool)
                sim_local_1, sim_local_2, sim_global, _ = model.forward_sim(im, ca, indices, t_mask, global_feats, current_validate_turns)
                # Only calculate the similarity of current validation rounds
                if sim_local_1.dim()==3:
                    sim_local_1 = sim_local_1[:, :, -1] 
                else:
                    pass
                if sim_local_2.dim()==3:
                    sim_local_2 = sim_local_2[:, :, -1]
                else:
                    pass
                sim_all = (sim_local_1 + sim_local_2 + sim_global) / 3              

            sims[im_start:im_end, ca_start:ca_end] = sim_all.data.cpu().numpy()
    sys.stdout.write('\n')
    return sims

def shard_attn_scores_test(model, img_embs, cap_embs, opt, indices, t_mask_all, global_feats, current_test_turns, shared_size=200):
    n_im_shard = len(img_embs) // shared_size
    n_cap_shard = len(cap_embs) // shared_size

    sims = np.zeros((9800, 9800))
    with torch.no_grad():
        for i in range(n_im_shard):
            im_start, im_end = shared_size * i, min(shared_size * (i + 1), len(img_embs))
            for j in range(n_cap_shard):
                sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
                ca_start, ca_end = shared_size * j, min(shared_size * (j + 1), len(cap_embs))

                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                t_mask = t_mask_all[ca_start:ca_end].to(torch.bool)
                sim_local_1, sim_local_2, sim_global, _ = model.forward_sim(im, ca, indices, t_mask, global_feats, current_test_turns)
                # Only calculate the similarity of current testing rounds
                if sim_local_1.dim()==3:
                    sim_local_1 = sim_local_1[:, :, -1] 
                else:
                    pass
                if sim_local_2.dim()==3:
                    sim_local_2 = sim_local_2[:, :, -1]
                else:
                    pass
                sim_all = (sim_local_1 + sim_local_2 + sim_global) / 3    

                sims[im_start:im_end, ca_start:ca_end] = sim_all.data.cpu().numpy()
    sys.stdout.write('\n')
    return sims

def eval_val(images, captions, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    sims: (N, N) matrix of similarity im-cap
    """
    npts = captions.shape[0]                                                          
    ranks = np.zeros(npts)                                                              
    top1 = np.zeros(npts)                                                               

    # --> (N(caption), N(image))
    sims = sims.T

    for index in range(npts):                                                           
        inds = np.argsort(sims[index])[::-1]                                           
        # Score
        rank = np.where(inds == index)[0][0]                                            
        ranks[index] = rank                                                            
        top1[index] = inds[0]                                                           

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, meanr)

def eval_test(images, captions, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    sims: (N, N) matrix of similarity im-cap
    """
    npts = captions.shape[0] - 96                                                       
    ranks = np.zeros(npts)                                                              
    top1 = np.zeros(npts)                                                               

    # --> (N(caption), N(image))
    sims = sims.T

    for index in range(npts):                                                           
        inds = np.argsort(sims[index])[::-1]                                            
        # Score
        rank = np.where(inds == index)[0][0]                                           
        ranks[index] = rank                                                           
        top1[index] = inds[0]                                                          

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, meanr)

if __name__ == '__main__':
    evalrank("./runs/vg/checkpoint/model_best.pth.tar", split="test")
