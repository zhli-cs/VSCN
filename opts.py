"""Argument parser"""

import argparse


def str2bool(v):
    return v.lower() in ('true', '1')

def parse_opt():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # --------------------------- data path -------------------------#
    parser.add_argument('--data_path', default='./data',
                        help='path to datasets')
    parser.add_argument('--model_save', default='./runs/vg/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--logger_path', default='./runs/vg/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to load model')    

    # ----------------------- training setting ----------------------#
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--learning_rate', default=2e-4, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=50, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=100, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--pin_memory', type=str2bool, default=True)  
    parser.add_argument('--seed', type=int, default=1)              

    # ------------------------- model setting -----------------------#
    parser.add_argument('--max_turns', default=10, type=int,           
                        help='max rounds for caption.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--embed_size', default=256, type=int,
                        help='Dimensionality of the joint embedding.')   
    parser.add_argument('--v_SA_dropout', default=0.4, type=float,
                        help="Dropout applied in the VisualSA")
    parser.add_argument('--t_SA_dropout', default=0.4, type=float,
                        help="Dropout applied in the TextualSA")
    parser.add_argument('--tau', type=float, default=15, help='Temperature coefficient of InfoNCE loss')          
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in transformer blocks")
    parser.add_argument('--v_dropout', default=0.1, type=float,
                        help="Dropout applied in transformers")
    parser.add_argument('--nhead', default=8, type=int,
                        help="Number of attention heads inside transformers")
    parser.add_argument('--v_transformer_layer', default=3, type=int,
                        help='Number of the visual context transformer layers.')
    parser.add_argument('--warmup_Eiters', type=int, default=2100) 
    parser.add_argument('--sample_option', type=str2bool, default=True,
                        help='whether to conduct query sampling') 
    parser.add_argument('--dropped_ratio', default=0.1, type=float,
                        help="dropped probability of queries")
    parser.add_argument('--vl_transformer_layer', default=3, type=int,
                        help='Number of the cross-modal context transformer layers.')
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the cross-modal context transformer")
    parser.add_argument('--loss_weight', default=1, type=float,
                        help="weight to balance the InfoNCE loss and the distillation loss")
    opt = parser.parse_args()
    return opt
