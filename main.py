import torch
from torch.utils.data import DataLoader, random_split
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs

from vqgan_vae import VQGanVAE
from trainer import VQGanVAETrainer, MaskGitTrainer
from scRNASeqEmbed import scRNASeqEmbedding
from muse_maskgit_SSST import MaskGit, MaskGitTransformer
from dataloader import scRNAseqSTDataset, get_num_cells

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"



torch.cuda.empty_cache()

#----------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------#
## Step 1

# # train VAE first
# # tried channel = 1 and channel =  3
# # Need to MODIFY 'VQGanVAE' in vqgan_vae.py if want to change to channel = 3 (channel = 3 is simply repeat the 1 channel)
# # 14249 is the num_cells

# vae = VQGanVAE(
#     dim = 256,
#     vq_codebook_size=14249 
# )

# trainer = VQGanVAETrainer(
#     vae=vae,
#     image_size = 512,
#     folder='../data/Dataset4',
#     batch_size=4,
#     grad_accum_every= 8,
#     num_train_steps=10000,
# ).cuda()

# trainer.train()

#----------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------#
## Step 2

# # Load the pre-trained VAE
# vae = VQGanVAE(
#     dim = 256,
#     vq_codebook_size=14249
# )
# # load vae
# vae.load('results_vqsize_numcells/vae.8000.pt')
# # set data directory
# data_dir='../data/Dataset4/'
# # get num cells
# num_cells = get_num_cells(data_dir)

# # (1) create transformer / attention network

# transformer = MaskGitTransformer(
#     num_tokens = num_cells,         # must be same as codebook size above
#     seq_len = 128 ** 2,            # must be equivalent to fmap_size ** 2 in vae
#     dim = 512,                # model dimension
#     depth = 8,                # depth
#     dim_head = 64,            # attention head dimension
#     heads = 8,                # attention heads,
#     ff_mult = 4,              # feedforward expansion factor
# )

# maskgit_trainer = MaskGitTrainer(
#     vae=vae,
#     transformer=transformer,
#     folder = data_dir,
#     num_train_steps=10000,
#     batch_size=4,
#     image_size=256,
#     cond_drop_prob=0.25,
# ).cuda()

# maskgit_trainer.train()


#----------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------#
## Step 3

# Load the pre-trained VAE
vae = VQGanVAE(
    dim = 256,
    vq_codebook_size=14249
)
# load vae
vae.load('results_vqsize_numcells/vae.8000.pt')
# set data directory
data_dir='../data/Dataset4/'
# get num cells
num_cells = get_num_cells(data_dir)


transformer = MaskGitTransformer(
    num_tokens = num_cells,         # must be same as codebook size above
    seq_len = 128 ** 2,           # must be equivalent to fmap_size ** 2 in vae
    dim = 512,                # model dimension
    depth = 2,                # depth
    dim_head = 64,            # attention head dimension
    heads = 8,                # attention heads,
    ff_mult = 4,              # feedforward expansion factor
)

superres_maskgit_trainer = MaskGitTrainer(
    vae=vae,
    transformer=transformer,
    folder = data_dir,
    is_superres=True,
    num_train_steps=10000,
    batch_size=4,
    image_size=512,
    cond_image_size = 256,
    cond_drop_prob=0.25,
    results_folder='./Superres_maskgit_result'
).cuda()

superres_maskgit_trainer.train()


#----------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------#
## Step 4

## Need new set up but should follow the original way
# 
from muse_maskgit_SSST import Muse
# Initialize before loading
# ... initialize transformer
# base_maskgit = MaskGit(
            #     vae = vae,                 # vqgan vae
            #     transformer = transformer, # transformer
            #     image_size = image_size,          # image size
            #     cond_drop_prob = cond_drop_prob,     # conditional dropout, for classifier free guidance
            # ).cuda()

## same with superres_maskgit. See MaskGitTrainer class in trainer.py. Should be between line 523 and 539
base_maskgit.load('./path/to/base.pt')

superres_maskgit.load('./path/to/superres.pt')

# pass in the trained base_maskgit and superres_maskgit from above

muse = Muse(
    base = base_maskgit,
    superres = superres_maskgit
)
# Change Muse class in muse_maskgit_SSST so that it should work in the same way we feed in our data when training the data. Return spatial images
images = muse([
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles',
    'waking up to a psychedelic landscape'
])

images # should be spatial images

