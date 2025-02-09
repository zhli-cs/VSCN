import torch
import torch.nn as nn
import torch.nn.functional as F

class ComputeFinalLoss(nn.Module):
    """
    Compute final loss = InfoNCE loss + distillation loss
    """
    def __init__(self, opt):
        super(ComputeFinalLoss, self).__init__()   
        self.infonce_loss = infoNCELoss(tau=opt.tau)
        self.loss_weight = opt.loss_weight

    def forward(self, sim_local_1, sim_local_2, sim_global, loss_dist):                                            

        bsize, bsize, current_turns = sim_local_2.size()

        # 1. compute infoNCE loss 1  
        loss_local_infoNCE_1 = [self.infonce_loss(sim_local_1[:, :, 0])]   
        if current_turns > 1:                                                  
            for i in range(1, current_turns):
                loss_local_infoNCE_1.append(self.infonce_loss(sim_local_1[:, :, i]))                           
        loss_local_infoNCE_1 = torch.stack(loss_local_infoNCE_1, -1)                                                                                
        loss_local_infoNCE_1 = torch.mean(loss_local_infoNCE_1, -1)                              

        # 2. compute infoNCE loss 2
        loss_local_infoNCE_2 = [self.infonce_loss(sim_local_2[:, :, 0])]   
        if current_turns > 1:                                                  
            for i in range(1, current_turns):
                loss_local_infoNCE_2.append(self.infonce_loss(sim_local_2[:, :, i]))                           
        loss_local_infoNCE_2 = torch.stack(loss_local_infoNCE_2, -1)                                                                                
        loss_local_infoNCE_2 = torch.mean(loss_local_infoNCE_2, -1)                              

        # 3. compute global infoNCE loss
        loss_global_infoNCE = self.infonce_loss(sim_global)                              

        # sum infoNCE loss
        loss_infoNCE = loss_local_infoNCE_1 + loss_local_infoNCE_2 + loss_global_infoNCE  

        # FINAL LOSS
        loss_all = loss_infoNCE + self.loss_weight * loss_dist 

        return loss_all, loss_infoNCE

class infoNCELoss(nn.Module):
    """
    Compute infoNCE loss
    """
    def __init__(self, tau=1):
        super(infoNCELoss, self).__init__()
        self.tau = tau

    def forward(self, scores):                                            

        bsize, bsize = scores.size()
        scores = self.tau * scores.clamp(min=-1e10)
        d1 = F.log_softmax(scores, dim=1)                                  
        d2 = F.log_softmax(scores, dim=0)                                  

        loss_s = torch.sum(d1.diag())
        loss_im = torch.sum(d2.diag())
        loss_infoNCE = -1 * (loss_s + loss_im) / bsize                     

        return loss_infoNCE                                                                      