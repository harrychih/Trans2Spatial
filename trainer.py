from math import sqrt
from random import choice
from pathlib import Path
from shutil import rmtree
from functools import partial

from beartype import beartype

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

from vqgan_vae import VQGanVAE
from muse_maskgit_SSST import MaskGit, MaskGitTransformer
from dataloader import scRNAseqSTDataset

import pyspark
import numpy as np

from einops import rearrange

from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs

from ema_pytorch import EMA

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# helper functions

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def noop(*args, **kwargs):
    pass

def find_index(arr, cond):
    for ind, el in enumerate(arr):
        if cond(el):
            return ind
    return None

def find_and_pop(arr, cond, default = None):
    ind = find_index(arr, cond)

    if exists(ind):
        return arr.pop(ind)

    if callable(default):
        return default()

    return default

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def scale_coordinates(coords, target_size):
    x_coords, y_coords = zip(*coords)

    min_x, min_y = min(x_coords), min(y_coords)
    max_x, max_y = max(x_coords), max(y_coords)

    def scale_value(val, min_val, max_val, target_size):
        return int(((val - min_val) / (max_val - min_val)) * (target_size - 1))

    scaled_coords = [(scale_value(x, min_x, max_x, target_size),
                     scale_value(y, min_y, max_y, target_size))
                     for x, y in coords]

    return scaled_coords


# image related helpers fnuctions and dataset

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        # exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.Insitu_count_path = f'{folder}/Insitu_count.txt'
        self.Locations_path = f'{folder}/Locations.txt'

        from pyspark.sql import SparkSession
        # Initialize the Spark session
        spark = SparkSession.builder \
            .appName("Large Text Files Reader") \
            .config('spark.port.maxRetries', 100) \
            .getOrCreate()

        
        self.insitu_count_df = spark.read.csv(self.Insitu_count_path, sep="\t", header=True)
        self.locations_df = spark.read.csv(self.Locations_path, sep="\t", header=True)

        print(f'{len(self.insitu_count_df.columns)} gene training samples found at {folder}')

        # Convert the locations_df DataFrame to a list of tuples
        self.locations = self.locations_df.rdd.map(lambda row: (float(row[0]), float(row[1]))).collect()

        # Scale the locations to fit a image-like system
        self.scaled_locations = scale_coordinates(self.locations, image_size)

        import numpy as np

        # Create an empty 3D array with dimensions (512, 512, num_genes)
        self.num_genes = len(self.insitu_count_df.columns)
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.insitu_count_df.columns)}
        spatial_image = np.zeros((image_size, image_size, self.num_genes))

        # create pandas dataframes
        self.insitu_count = self.insitu_count_df.toPandas()

        # iterate through the rows of the insitu_count dataframe, and fill the spatial_image array
        for index, row in self.insitu_count.iterrows():
            for gene in self.insitu_count_df.columns:
                spatial_image[self.scaled_locations[index][0], self.scaled_locations[index][1], self.gene_to_idx[gene]] = row[gene]

        self.spatial_images = torch.from_numpy(np.float32(spatial_image))

    def __len__(self):
        return self.num_genes

    def __getitem__(self, index):
        return self.spatial_images[:,:,index].unsqueeze(dim=0)# .repeat(3, 1, 1) # For 3 channels, repeat

# main trainer class

@beartype
class VQGanVAETrainer(nn.Module):
    def __init__(
        self,
        vae: VQGanVAE,
        *,
        folder,
        num_train_steps,
        batch_size,
        image_size,
        lr = 3e-4,
        grad_accum_every = 1,
        max_grad_norm = None,
        discr_max_grad_norm = None,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        valid_frac = 0.05,
        random_split_seed = 3407,
        use_ema = True,
        ema_beta = 0.995,
        ema_update_after_step = 0,
        ema_update_every = 1,
        apply_grad_penalty_every = 4,
        accelerate_kwargs: dict = dict(),
    ):
        super().__init__()

        # instantiate accelerator

        kwargs_handlers = accelerate_kwargs.get('kwargs_handlers', [])

        ddp_kwargs = find_and_pop(
            kwargs_handlers,
            lambda x: isinstance(x, DistributedDataParallelKwargs),
            partial(DistributedDataParallelKwargs, find_unused_parameters = True)
        )

        ddp_kwargs.find_unused_parameters = True
        kwargs_handlers.append(ddp_kwargs)
        accelerate_kwargs.update(kwargs_handlers = kwargs_handlers)

        self.accelerator = Accelerator(**accelerate_kwargs)

        # vae

        self.vae = vae

        # training params

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        all_parameters = set(vae.parameters())
        # discr_parameters = set(vae.discr.parameters())
        vae_parameters = all_parameters # - discr_parameters

        self.vae_parameters = vae_parameters

        # optimizers

        self.optim = Adam(vae_parameters, lr = lr)
        # self.discr_optim = Adam(discr_parameters, lr = lr)

        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm

        # create dataset

        self.ds = ImageDataset(folder, image_size)

        # split for validation

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = DataLoader(
            self.ds,
            batch_size = batch_size,
            shuffle = True
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size = batch_size,
            shuffle = True
        )

        # prepare with accelerator

        (
            self.vae,
            self.optim,
            # self.discr_optim,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.vae,
            self.optim,
            # self.discr_optim,
            self.dl,
            self.valid_dl
        )

        self.use_ema = use_ema

        if use_ema:
            self.ema_vae = EMA(vae, update_after_step = ema_update_after_step, update_every = ema_update_every)
            self.ema_vae = self.accelerator.prepare(self.ema_vae)

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.apply_grad_penalty_every = apply_grad_penalty_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model = self.accelerator.get_state_dict(self.vae),
            optim = self.optim.state_dict(),
            discr_optim = self.discr_optim.state_dict()
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        vae = self.accelerator.unwrap_model(self.vae)
        vae.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])
        self.discr_optim.load_state_dict(pkg['discr_optim'])

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())
        apply_grad_penalty = not (steps % self.apply_grad_penalty_every)

        self.vae.train()
        discr = self.vae.module.discr if self.is_distributed else self.vae.discr
        if self.use_ema:
            ema_vae = self.ema_vae.module if self.is_distributed else self.ema_vae

        # logs

        logs = {}

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            img = next(self.dl_iter)
            img = img.to(device)

            with self.accelerator.autocast():
                loss = self.vae(
                    img,
                    add_gradient_penalty = apply_grad_penalty,
                    return_loss = True
                )

            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.vae.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # update discriminator

        if exists(discr):
            self.discr_optim.zero_grad()

            for _ in range(self.grad_accum_every):
                img = next(self.dl_iter)
                img = img.to(device)

                loss = self.vae(img, return_discr_loss = True)

                self.accelerator.backward(loss / self.grad_accum_every)

                accum_log(logs, {'discr_loss': loss.item() / self.grad_accum_every})

            if exists(self.discr_max_grad_norm):
                self.accelerator.clip_grad_norm_(discr.parameters(), self.discr_max_grad_norm)

            self.discr_optim.step()

            # log

            self.print(f"{steps}: vae loss: {logs['loss']} - discr loss: {logs['discr_loss']}")

        self.print(f"{steps}: vae loss: {logs['loss']}")

        # update exponential moving averaged generator

        if self.use_ema:
            ema_vae.update()

        # sample results every so often

        if not (steps % self.save_results_every):
            vaes_to_evaluate = ((self.vae, str(steps)),)

            if self.use_ema:
                vaes_to_evaluate = ((ema_vae.ema_model, f'{steps}.ema'),) + vaes_to_evaluate

            for model, filename in vaes_to_evaluate:
                model.eval()

                valid_data = next(self.valid_dl_iter)
                valid_data = valid_data.to(device)

                recons = model(valid_data, return_recons = True)

                # else save a grid of images

                imgs_and_recons = torch.stack((valid_data, recons), dim = 0)
                imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

                imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
                grid = make_grid(imgs_and_recons, nrow = 2, normalize = True, value_range = (0, 1))

                logs['reconstructions'] = grid

                save_image(grid, str(self.results_folder / f'{filename}.png'))

            self.print(f'{steps}: saving to {str(self.results_folder)}')

        # save model every so often
        self.accelerator.wait_for_everyone()
        if self.is_main and not (steps % self.save_model_every):
            state_dict = self.accelerator.unwrap_model(self.vae).state_dict()
            model_path = str(self.results_folder / f'vae.{steps}.pt')
            self.accelerator.save(state_dict, model_path)

            if self.use_ema:
                ema_state_dict = self.accelerator.unwrap_model(self.ema_vae).state_dict()
                model_path = str(self.results_folder / f'vae.{steps}.ema.pt')
                self.accelerator.save(ema_state_dict, model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def train(self, log_fn = noop):
        device = next(self.vae.parameters()).device

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')



@beartype
class MaskGitTrainer(nn.Module):
    def __init__(
        self,
        vae: VQGanVAE,
        transformer: MaskGitTransformer,
        *,
        folder,
        num_train_steps,
        batch_size,
        image_size,
        lr = 3e-4,
        cond_drop_prob = 0.25,
        is_superres = False,
        cond_image_size = None,
        save_results_every = 100,
        save_model_every = 100,
        results_folder = './MaskGit_result',
        max_grad_norm = None,
        valid_frac = 0.05,
        random_split_seed = 3407,
        apply_grad_penalty_every = 4,
        accelerate_kwargs: dict = dict(),
    ):
        super().__init__()

        # instantiate accelerator

        kwargs_handlers = accelerate_kwargs.get('kwargs_handlers', [])

        ddp_kwargs = find_and_pop(
            kwargs_handlers,
            lambda x: isinstance(x, DistributedDataParallelKwargs),
            partial(DistributedDataParallelKwargs, find_unused_parameters = True)
        )

        ddp_kwargs.find_unused_parameters = True
        kwargs_handlers.append(ddp_kwargs)
        accelerate_kwargs.update(kwargs_handlers = kwargs_handlers)

        self.accelerator = Accelerator(**accelerate_kwargs)

        # save image size for later use
        self.image_size = image_size


        # training params

        self.register_buffer('steps', torch.Tensor([0]))
        self.register_buffer('epoch', torch.Tensor([1]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm


        # maskgit initialize
        if not is_superres:
            self.maskgit = MaskGit(
                vae = vae,                 # vqgan vae
                transformer = transformer, # transformer
                image_size = image_size,          # image size
                cond_drop_prob = cond_drop_prob,     # conditional dropout, for classifier free guidance
            ).cuda()
        else:
            assert exists(cond_image_size), 'conditional image size must be given if training the superres Maskgit!'
            self.maskgit = MaskGit(
                vae = vae,
                transformer = transformer,
                cond_drop_prob = cond_drop_prob,
                image_size = image_size,                     # larger image size
                cond_image_size = cond_image_size,                # conditioning image size <- this must be set
            ).cuda()

        # # optimizers

        self.optim = Adam(self.maskgit.parameters(), lr = lr)

        # create dataset

        self.ds = scRNAseqSTDataset(data_dir=folder, image_size=image_size)

        # split for validation

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = DataLoader(
            self.ds,
            batch_size = batch_size,
            shuffle = True
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size = batch_size,
            shuffle = True
        )

        # prepare with accelerator

        (
            self.maskgit,
            self.optim,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.maskgit,
            self.optim,
            self.dl,
            self.valid_dl
        )

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.check_val_step = 0

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.apply_grad_penalty_every = apply_grad_penalty_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model = self.accelerator.get_state_dict(self.maskgit),
            optim = self.optim.state_dict(),
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        maskgit = self.accelerator.unwrap_model(self.maskgit)
        maskgit.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())
        epoch = int(self.epoch.item())
        
        self.maskgit.train()

        # logs

        logs = {}

        # update maskgit

        text, img = next(self.dl_iter)
        text = text.to(device)
        img = img.to(device)

        with self.accelerator.autocast():
            loss = self.maskgit(
                img,
                texts = text
            )

        self.accelerator.backward(loss)

        logs[f"epoch {epoch}, step {steps}"] = loss.item()

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.maskgit.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        val_losses = []
        val_loss = 0
        self.check_val_step += self.batch_size

        if self.check_val_step >= len(self.dl):
            self.maskgit.eval()
            for text, img in self.valid_dl:
                text, img = text.to(device), img.to(device)
                with self.accelerator.autocast():
                    val_losses.append(self.maskgit(
                        img,
                        texts = text
                    ).item())
            val_loss = np.mean(val_losses)
            self.check_val_step = 0
            self.print(f"Epoch {epoch} - [{steps}/{self.num_train_steps}] - loss: {loss.item()} - val_loss: {val_loss}")
            self.epoch += 1
        else:
            self.print(f"Epoch {epoch} - [{steps}/{self.num_train_steps}]: loss: {loss.item()}")

        # save model every so often
        self.accelerator.wait_for_everyone()
        if self.is_main and not (steps % self.save_model_every):
            state_dict = self.accelerator.unwrap_model(self.maskgit).state_dict()
            model_path = str(self.results_folder / f'maskgit_{self.image_size}.{steps}.pt')
            self.accelerator.save(state_dict, model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def train(self, log_fn = noop):
        device = next(self.maskgit.parameters()).device

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')

    def retrain(self, model_dir, log_fn = noop):
        self.load(model_dir)
        self.train()
