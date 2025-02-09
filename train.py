import logging
import os
import pickle
import random
import shutil
import time

import numpy as np
import tensorboard_logger as tb_logger
import torch

import data
import opts
from evaluation import (AverageMeter, LogCollector, encode_data, shard_attn_scores_val, eval_val)
from model import VSCN
from vg import vg
from vocab import Vocabulary

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    opt = opts.parse_opt()
    print(opt)
    # LOG
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logger_path = os.path.abspath(opt.logger_path)
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)
    tb_logger.configure(opt.logger_path, flush_secs=5)
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())  

    # random seed
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    
    # Load dataset
    train_db = vg(opt, 'train')
    val_db   = vg(opt, 'val')

    # Load data loaders
    train_loader, val_loader = data.get_loaders(train_db, val_db, opt.batch_size, opt.workers, opt.pin_memory)

    # Construct the model
    model = VSCN(opt)

    # load model and options, continue training
    if opt.model_path is not None:
        checkpoint = torch.load(opt.model_path)
        # opt = checkpoint['opt']
        save_epoch = checkpoint['epoch']
        best_rsum = checkpoint['best_rsum']
        print(opt)
        # load model state
        model.load_state_dict(checkpoint['model'])

    # Train the Model
    best_rsum = 0

    for epoch in range(opt.num_epochs):
        print(opt.logger_path)
        print(opt.model_save)

        # learning rate decay
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        r_sum = validate(opt, val_loader, model, epoch)

        # remember best R@ sum and save checkpoint
        is_best = r_sum > best_rsum
        best_rsum = max(r_sum, best_rsum)

        # save checkpoint
        model_save_path = os.path.abspath(opt.model_save)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_save + '/', time=t)


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                .format(
                    epoch, i, len(train_loader), e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

    # Validate at every val_step
        # if model.Eiters % opt.val_step == 0:
        #     validate(opt, val_loader, model, epoch)


def validate(opt, val_loader, model, epoch):
    '''
    Only evaluate the retrieval performance of the first round in validation.
    '''
    # Compute the encoding for all the validation images and captions
    with torch.no_grad(): 
        img_embs, cap_embs, global_feats = encode_data(model, val_loader, opt.log_step, logging.info)

        # Record computation time of validation
        start = time.time()
        #  Construct the query mask for the first round
        indices = None
        N = img_embs.shape[0]  
        t_mask_current_turn = torch.cat([torch.zeros(1), torch.ones(opt.max_turns-1)], dim=0)  
        t_mask_current_turn_all = t_mask_current_turn.repeat(N, 1).cuda()  
        sims = shard_attn_scores_val(model, img_embs, cap_embs, opt, indices, t_mask_current_turn_all, global_feats, current_validate_turns=1, shared_size=250)
        end = time.time()
        print("calculate similarity time:", end-start)

        # Image retrieval
        (r1i, r5i, r10i, meanr) = eval_val(img_embs, cap_embs, sims)
        logging.info("1st round image retrieval: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, meanr))

        # Sum of recalls to be used for early stopping
        r_sum = r1i + r5i + r10i                                                                       
    
    torch.cuda.empty_cache()

    # Record metrics in tensorboard
    tb_logger.log_value('r1_t2i', r1i, step=epoch)
    tb_logger.log_value('r5_t2i', r5i, step=epoch)
    tb_logger.log_value('r10_t2i', r10i, step=epoch)
    tb_logger.log_value('meanr_t2i', meanr, step=epoch)
    tb_logger.log_value('r_sum', r_sum, step=epoch)

    return r_sum


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix='', time=''):
    if is_best:
        torch.save(state, prefix + time + ' model_best.pth.tar')

def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR
    multiplies 0.1 after every 15 epoch
    """
    if epoch > 0 and epoch % opt.lr_update == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

if __name__ == '__main__':
    main()
