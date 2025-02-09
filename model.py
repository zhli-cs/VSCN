from collections import OrderedDict
from turtle import forward

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from compute_loss import ComputeFinalLoss
from GPO import GPO
from visual_context_transformer import VisualContextTransformerEncoderLayer
from crossmodal_context_transformer import CrossModalContextTransformerEncoderLayer

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        if x.dim() == 2:
            for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
                x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        elif x.dim() == 3:
            B, N, D = x.size()
            x = x.reshape(B*N, D)
            for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
                x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
            x = x.view(B, N, self.output_dim)
        return x

class EncoderImage(nn.Module):
    """
    Build local region representations.
    """
    def __init__(self, opt, img_dim, embed_size):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        img_emb = self.fc(images)
        img_emb = l2norm(img_emb, dim=-1)

        return img_emb

    def load_state_dict(self, state_dict):
        """Overwrite the default one to accept state_dict from Full model"""
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)

class EncoderText(nn.Module):
    """
    Sample queries and build local query representations.
    """
    def __init__(self, opt, embed_size):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.sample_option = opt.sample_option
        self.dropped_ratio = opt.dropped_ratio

        self.project = nn.Linear(512, opt.embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.project.in_features +
                                  self.project.out_features)
        self.project.weight.data.uniform_(-r, r)
        self.project.bias.data.fill_(0)

    def forward(self, captions):      
        bsize, max_turns, emded_size = captions.size()
        # Query Sampler
        if self.training:
            # Whether to conduct query sampling
            if self.sample_option:  
                num_query = captions.shape[1]
                rand_list = np.random.rand(num_query)
                ind = np.where(rand_list > self.dropped_ratio)[0]
                indices = torch.tensor(ind).cuda()
                # Identify which rounds of the selected queries in this batch are involved in training. 
                # A value of '1' in the t_mask indicates this round is hidden and does not participate in training. 
                # A value of '0' indicates this round is included in training.
                t_mask = torch.ones(max_turns)                  
                if indices.numel() == 0:
                    # If no query is sampled, randomly select one round from the ten rounds.
                    id = np.random.permutation(range(max_turns))[0]
                    t_mask[id] = 0
                    indices = torch.tensor(id).cuda()
                else:
                    for i in range(len(ind)):
                        select_ind = ind[i]
                        t_mask[select_ind] = 0
                new_t_mask = t_mask.repeat(bsize, 1).cuda()    
            else:
                t_mask = torch.zeros(max_turns)
                new_t_mask = t_mask.repeat(bsize, 1).cuda()
                indices = torch.arange(0, 10, 1).cuda()
        else:
            # Determined in evaluation.py
            indices = None
            new_t_mask = None
 
        cap_emb = self.project(captions)
        cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, indices, new_t_mask                                                                                        

class VisualSA(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings.
          - raw_global: raw image by averaging regions.
    Returns: - new_global: final image by self-attention.
    """
    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.BatchNorm1d(num_region),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.BatchNorm1d(embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local regions and raw global image
        l_emb = self.embedding_local(local)                        
        g_emb = self.embedding_global(raw_global)                  

        # compute the normalized weights
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)    
        common = l_emb.mul(g_emb)                                
        weights = self.embedding_common(common).squeeze(2)        
        weights = self.softmax(weights)                            

        # compute final image
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)     
        new_global = l2norm(new_global, dim=-1)                    

        return new_global

class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings.
          - raw_global: raw text by averaging words.
    Returns: - new_global: final text by self-attention.
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)                         
        g_emb = self.embedding_global(raw_global)                   

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)      
        common = l_emb.mul(g_emb)                                   
        weights = self.embedding_common(common).squeeze(2)          
        weights = self.softmax(weights)                             

        # compute final text, shape: (batch_size, 256)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)      
        new_global = l2norm(new_global, dim=-1)                     

        return new_global

class EncoderSimilarity(nn.Module):
    """
    Args: - img_emb: local region embeddings.
          - cap_emb: local query embeddings.
          - indices: Indexes of sampled rounds.
          - t_mask: masks for sampled and unsampled queries in the cross-modal context transformer.
          - global_features: CLIP-encoded global image features.
          - current_test_turns: the number of rounds for validation or testing.

    Returns:
        - sim_local_1: context perception similarity S1, shape: (batch_size, batch_size, current_turns).
        - sim_local_2: context perception similarity S2, shape: (batch_size, batch_size, current_turns)
        - sim_global: context interaction similarity.
        - loss_dist: the distillation loss, shape: ()
    """
    def __init__(self, opt, embed_size, v_transformer_layer=6, vl_transformer_layer=6):
        super(EncoderSimilarity, self).__init__()
        self.opt = opt
        self.num_region = 36
        self.v_global_w = VisualSA(embed_size, opt.v_SA_dropout, self.num_region)                    
        self.v_transformer = nn.ModuleList([VisualContextTransformerEncoderLayer(opt, opt.embed_size, opt.nhead, opt.dim_feedforward, opt.v_dropout) for i in range(v_transformer_layer)])
        self.vl_transformer = nn.ModuleList([CrossModalContextTransformerEncoderLayer(opt, opt.embed_size, opt.nhead, opt.dim_feedforward, opt.vl_dropout) for i in range(vl_transformer_layer)])
        self.cls_token = nn.Embedding(1, embed_size)
        self.distill_loss = DistillLoss(opt)
        self.proj = MLP(512, embed_size, embed_size, 2)
        self.gpool = GPO(16, 16)

        self.init_weights()

    def compute_pairwise_similarity(self, src_feats, tgt_feats):
        sim = torch.bmm(tgt_feats, src_feats.transpose(1, 2))                                       
        sim = nn.LeakyReLU(0.1)(sim)  
        return sim  

    def pairwise_similarity_to_attn(self, pairwise_similarities):
        attn = pairwise_similarities.clamp(min=-1e10)                          
        attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]                                       
        attn = F.softmax(attn, dim=-1)                                                               
        return attn
 
    def forward(self, img_emb, cap_emb, indices, t_mask, global_feats, current_test_turns):     
        bsize, n_regions, embed_size = img_emb.size()

        # Determine whether it is in training mode or testing mode
        if self.training:
            # Sample queries available in each training batch
            cap_emb_select = torch.index_select(cap_emb, 1, indices)  
            current_turns = cap_emb_select.size(1)
        else:
            # Determine the number of queries during validation
            cap_emb_select = cap_emb[:, :current_test_turns, :]                                      
            current_turns = cap_emb_select.size(1)
        # =================================================================================== #
        #                           Context Semantic Perception                               #
        # =================================================================================== #
        # =================================================================================== #
        #            1. Intra-context Exploration with Visual Context Transformer             #
        # =================================================================================== #
        cls_emb = self.cls_token.weight.unsqueeze(1).repeat(bsize, 1, 1)                              
        src_emb = torch.cat([cls_emb, img_emb], dim=1)                                              
        for module in self.v_transformer:
            src_emb = module(src_emb, need_weights=False)                                             

        # Regional intra-context features
        img_emb = src_emb[:, 1:, :]                                                                 
        # Global intra-context features
        cls_emb = src_emb[:, 0, :]                                                                    

        # Compute context perception similarity S1
        region_feats = img_emb.view(1, bsize, n_regions, embed_size)
        region_feats = region_feats.expand(bsize, bsize, n_regions, embed_size).contiguous()
        region_feats = region_feats.view(bsize, bsize * n_regions, embed_size).contiguous()

        sim_local_1 = torch.zeros(bsize, bsize, current_turns).cuda()                           
        for i in range(current_turns):
            cap_emb_i = cap_emb_select[:, :i+1, :]                                                               
            sim_region = self.compute_pairwise_similarity(cap_emb_i, region_feats)                         
            attn_region = self.pairwise_similarity_to_attn(sim_region)       
            sim_CurrentRound_local_1 = torch.sum(sim_region * attn_region, dim=-1)                          
            sim_CurrentRound_local_1 = sim_CurrentRound_local_1.view(bsize, bsize, n_regions)                
            sim_CurrentRound_local_1 = torch.mean(sim_CurrentRound_local_1, -1)                   
            sim_local_1[:, :, i] = sim_CurrentRound_local_1

        # =================================================================================== #
        #         2. Inter-context Exploration with Cross-modal Context Transformer           #
        # =================================================================================== #

        # Fuse visual and linguistic tokens
        vl_src = torch.cat([img_emb, cap_emb], dim=1)
        # Regarding input masks, ''False'' indicates tokens remain visable during the cross-attention operation, while ''True'' indicates otherwise.
        t_mask = t_mask.to(torch.bool)
        v_mask = torch.zeros((bsize, img_emb.size(1))).cuda().to(torch.bool)     
        vl_mask = torch.cat([v_mask, t_mask], dim=1)
        for module in self.vl_transformer:
            vl_src = module(vl_src, vl_mask, need_weights=False)                                           
        v_src = vl_src[:, 0:(self.num_region), :]                                                       

        # Compute context perception similarity S2
        region_feats_2 = v_src.view(1, bsize, n_regions, embed_size)
        region_feats_2 = region_feats_2.expand(bsize, bsize, n_regions, embed_size).contiguous()
        region_feats_2 = region_feats_2.view(bsize, bsize * n_regions, embed_size).contiguous()

        sim_local_2 = torch.zeros(bsize, bsize, current_turns).cuda()                             
        for i in range(current_turns):
            cap_emb_i = cap_emb_select[:, :i+1, :]                                                               
            sim_region_2 = self.compute_pairwise_similarity(cap_emb_i, region_feats_2)                         
            attn_region_2 = self.pairwise_similarity_to_attn(sim_region_2)        
            sim_CurrentRound_local_2 = torch.sum(sim_region_2 * attn_region_2, dim=-1)                          
            sim_CurrentRound_local_2 = sim_CurrentRound_local_2.view(bsize, bsize, n_regions)               
            sim_CurrentRound_local_2 = torch.mean(sim_CurrentRound_local_2, -1)                             
            sim_local_2[:, :, i] = sim_CurrentRound_local_2                             
        
        # =================================================================================== #
        #                           Context Semantic Interaction                              #
        # =================================================================================== #

        # Only perform CLIP-guided knowledge distillation during training
        if self.training:
            global_feats_proj = self.proj(global_feats.cuda())
            loss_dist = self.distill_loss(cls_emb, global_feats_proj)
        else:
            loss_dist = None
        
        # Obtain enhanced global images by self-attention
        img_ave = cls_emb
        img_glo = self.v_global_w(v_src, img_ave)                                        

        # Obtain enhanced global captions by self-attention
        cap_glo, _ = self.gpool(cap_emb_select, current_turns)                                                   

        # Compute context interaction similarity
        sim_global = torch.mm(img_glo, cap_glo.t())                                                      

        return sim_local_1, sim_local_2, sim_global, loss_dist

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DistillLoss(nn.Module):
    """
    Compute distillation loss
    """
    def __init__(self, opt):
        super(DistillLoss, self).__init__()
        self.proj = nn.Linear(512, opt.embed_size)

    def forward(self, cls_emb, global_features_proj):                       
        loss_dist = 0
        if torch.is_tensor(global_features_proj) == True:                   
            global_features_proj = global_features_proj.cuda()
            loss_dist = F.l1_loss(cls_emb, global_features_proj)
        else:
            pass

        return loss_dist 
    
class VSCN(object):
    def __init__(self, opt):
        # Build models
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.loss_weight = opt.loss_weight
        self.img_enc = EncoderImage(opt, opt.img_dim, opt.embed_size)
        self.txt_enc = EncoderText(opt, opt.embed_size)
        self.sim_enc = EncoderSimilarity(opt, opt.embed_size,
                                        opt.v_transformer_layer, opt.vl_transformer_layer)
        self.compute_loss = ComputeFinalLoss(opt)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.sim_enc.cuda()
            cudnn.benchmark = True

        # Loss and optimizer
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.sim_enc.parameters())
        self.params = params    

        self.optimizer = torch.optim.AdamW(params, lr=opt.learning_rate)
        self.learning_rate = opt.learning_rate

        self.Eiters = 0
        self.warmup_Eiters = opt.warmup_Eiters

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.sim_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.sim_enc.load_state_dict(state_dict[2])

    def train_start(self):
        """switch to train mode"""
        self.img_enc.train()
        self.txt_enc.train()
        self.sim_enc.train()

    def val_start(self):
        """switch to evaluate mode"""
        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_enc.eval()

    def forward_emb(self, images, captions):
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward feature encoding                                                                    
        img_embs = self.img_enc(images)                                                                    
        cap_embs, indices, t_mask = self.txt_enc(captions)                                                
        return img_embs, cap_embs, indices, t_mask                                                         

    def forward_sim(self, img_embs, cap_embs, indices, t_mask, global_feats, current_test_turns):
        # Forward similarity encoding
        sim_local_1, sim_local_2, sim_global, loss_dist = self.sim_enc(img_embs, cap_embs, indices, t_mask, global_feats, current_test_turns)
        return sim_local_1, sim_local_2, sim_global, loss_dist                                                             

    def forward_loss(self, sim_local_1, sim_local_2, sim_global, loss_dist, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        bsize = sim_global.size(0) 
        loss, loss_infoNCE = self.compute_loss(sim_local_1, sim_local_2, sim_global, loss_dist)
        self.logger.update('infoNCE', loss_infoNCE.item(), bsize)
        self.logger.update('distill', self.loss_weight * loss_dist.item(), bsize)
        self.logger.update('Loss', loss.item(), bsize)
        return loss

    def train_emb(self, image_feats, caption_feats, global_feats, ids, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        # Warmup
        if self.Eiters <= self.warmup_Eiters:
            warmup_percent_done = self.Eiters / self.warmup_Eiters
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_percent_done * self.learning_rate

        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # Compute the embeddings
        img_embs, cap_embs, indices, t_mask = self.forward_emb(image_feats, caption_feats)
        sim_local_1, sim_local_2, sim_global, loss_dist = self.forward_sim(img_embs, cap_embs, indices, t_mask, global_feats, current_test_turns=1)   

        # Measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(sim_local_1, sim_local_2, sim_global, loss_dist)                            

        # Compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()  

