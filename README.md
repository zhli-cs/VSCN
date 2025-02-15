# Visual Semantic Contextualization Network for Multi-Query Image Retrieval
PyTorch implementation for IEEE TMM 2025 paper “Visual Semantic Contextualization Network for Multi-Query Image Retrieval” by **Zhong Ji***, **Zhihao Li***, Yan Zhang, Yanwei Pang, Xuelong Li.
## Requirements
- Setup a conda environment and install some prerequisite packages including
```
conda create -n VSCN python=3.8    # Create a virtual environment
conda activate VSCN         	   # Activate virtual environment
conda install jupyter scikit-image cython opencv seaborn nltk pycairo h5py  # Install dependencies
python -m nltk.downloader all	   # Install NLTK data
```
- Please also install PyTorch 2.0.1 (or higher), torchvision and torchtext.
## Data
Refer [DrillDown](https://github.com/uvavision/DrillDown) to download images, features and annotations of Visual Genome.

Download `DrillDown/data/caches` from [DrillDown](https://github.com/uvavision/DrillDown) and put the directory under `VSCN/data`.

Download [CLIP-encoded visual and textual features](https://drive.google.com/drive/folders/1GySNYatVjhx5EJl-EnyIojmD90Ibs4bn?usp=sharing) of Visual Genome.

Put all data under the directory `VSCN/data` and organize them as follows:
```
data
├── caches
│   ├── raw_test.txt 
│   ├── vg_attributes_vocab_1000.txt
│   ├── vg_objects_vocab_1600.txt 
│   ├── vg_objects_vocab_2500.txt 
│   ├── vg_relations_vocab_500.txt 
│   ├── vg_scenedb.pkl                   # auto-generated upon initial execution
│   ├── vg_test.txt 
│   ├── vg_train.txt 
│   ├── vg_val.txt 
│   ├── vg_vocab_14284.pkl  
│   
├── vg
│   ├── caption_embedding_clip_vit_base16   
│   │      ├── test  
│   │           ├── xxx.npy
│   │           └── ...
│   │      ├── train 
│   │           ├── xxx.npy
│   │           └── ...
│   │      ├── val   
│   │           ├── xxx.npy
│   │           └── ...
│   ├── global_features 
│   │      ├── xxx.npz
│   │      └── ... 
│   ├── region_36_final   
│   │      ├── xxx.npz
│   │      └── ...
│   ├── visual_embedding_clip_vit_base16   
│   │      ├── xxx.npy
│   │      └── ... 
│   └── rg_jsons 
│   │      ├── xxx.json
│   │      └── ... 
│   └── sg_xmls
│   │      ├── xxx.xml
│   │      └── ... 
│   └── VG_100K
│   │      ├── xxx.jpg
│   │      └── ...
│   └── VG_100K_2
│   │      ├── xxx.jpg
│   │      └── ...
│
```
## Training
- Train VSCN
```
python train.py --max_turns 10 --sample_option True --dropped_ratio 0.1
```
## Evaluation
Please rename the saved checkpoint (the one with the best performance on the validation set) as `model_best.pth.tar`, then run the evaluation script.

- Evaluate VSCN
```
python evaluation.py
```
## Acknowledgment
This codebase is partially based on [DrillDown](https://github.com/uvavision/DrillDown) and [HMRN](https://github.com/zhli-cs/HMRN).

<!-- ## Citation
If you find our paper/code useful, please cite the following paper:
```
TO be finished...
``` -->
