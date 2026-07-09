# Trans2Spatial

**Translating single-cell RNA-seq into spatial transcriptomics with masked generative transformers.**

Trans2Spatial is a generative framework that learns to synthesize spatial transcriptomic *images* — gene-resolved 2D maps of in-situ expression — conditioned on single-cell RNA-seq (scRNA-seq) count vectors. It adapts the [Muse / MaskGit](https://github.com/lucidrains/muse-maskgit-pytorch) masked-token image-generation paradigm to the transcriptomics domain: a VQGAN-VAE first discretizes spatial expression maps into a codebook of cell-level tokens, after which a pair of masked-prediction transformers (a base generator and a super-resolution generator) autoregressively denoise a fully masked token canvas, conditioned on scRNA-seq embeddings, into a high-resolution spatial map.

> Code adapted from [lucidrains/muse-maskgit-pytorch](https://github.com/lucidrains/muse-maskgit-pytorch).

---

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Data Format](#data-format)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration & Key Hyperparameters](#configuration--key-hyperparameters)
- [Module Reference](#module-reference)
- [Notes & Caveats](#notes--caveats)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

Spatial transcriptomics measures gene expression while preserving tissue location, but it is expensive and low-throughput. scRNA-seq, by contrast, is high-throughput but loses spatial context. **Trans2Spatial** bridges this gap by *predicting* spatial expression maps from scRNA-seq profiles, effectively "imputing" a spatial arrangement onto dissociated single-cell data.

The model treats each gene's spatial expression as a single-channel image whose pixel values are expression counts laid out at the corresponding tissue coordinates. Generation then proceeds in two transformer stages — a **base** MaskGit producing a low-resolution token map, and a **super-resolution** MaskGit upsampling it to a high-resolution map — orchestrated by a `Muse` wrapper. Conditioning is supplied by an embedding of the scRNA-seq count vector, injected via cross-attention.

### Pipeline

```
                         ┌─────────────────────────────────────────────┐
   Spatial map (image)   │                  VQGAN-VAE                  │   Discrete codes
   (per-gene, 1 channel) ─►  ResnetEncDec ──►  Vector Quantization  ──►  (token indices)
                         └─────────────────────────────────────────────┘
                                    ▲                              │
                                    │ train (Step 1)               │
                                    │                              ▼
   scRNA-seq count ──► scRNASeqEmbedding ──► cross-attn ──►  Base MaskGit Transformer
                                                        (Step 2)        │
                                                                        ▼
                                                              low-res token map
                                                                        │
                              super-res cond ──────────────────► Super-Res MaskGit
                                                            (Step 3)        │
                                                                            ▼
                                                                high-res token map
                                                                        │
                                                                        ▼
                                                     VQGAN-VAE decode ──► Spatial image
                                                              (Step 4, via Muse)
```

## Methodology

1. **Spatial image construction.** In-situ counts at each tissue spot are binned onto an `image_size × image_size` grid by scaling the spot coordinates into `[0, image_size-1]`. The result is a 3D tensor `(H, W, num_genes)`; slicing along the gene axis yields one single-channel spatial "image" per gene, paired with that gene's scRNA-seq count vector.

2. **VQGAN-VAE (Step 1).** A ResNet encoder–decoder with a vector-quantization bottleneck learns to reconstruct spatial images. The codebook size is set to `num_cells` so each code corresponds to a cell identity, aligning the token space with the scRNA-seq vocabulary. Reconstruction uses L1 (or L2) loss; optional GAN + VGG perceptual losses are available via `use_vgg_and_gan` (off by default for the single-channel grayscale setting).

3. **Base MaskGit (Step 2).** A transformer with cross-attention to the scRNA-seq embedding is trained to predict masked VQ token ids via cross-entropy, following a cosine masking schedule. Classifier-free guidance (`cond_drop_prob`) and self-conditioning are supported.

4. **Super-resolution MaskGit (Step 3).** A second MaskGit, conditioned on the low-resolution token map (encoded by the VAE) *and* the scRNA-seq embedding, learns to upsample to the high-resolution token map.

5. **Muse inference (Step 4).** The trained base and super-resolution MaskGits are wrapped in a `Muse` module: the base generates a low-res token map, which the super-resolution stage refines into a high-res spatial image, decoded back through the VAE.

## Project Structure

```
Trans2Spatial/
├── main.py                  # End-to-end pipeline driver (Steps 1–4)
├── dataloader.py            # scRNAseqSTDataset: builds spatial images, pairs with scRNA counts
├── scRNAseqProcess.py       # Spark helpers for gene/index mapping
├── scRNASeqEmbed.py         # scRNA-seq count embedding (cell-index embedding + counts)
├── vqgan_vae.py             # VQGAN-VAE: encoder/decoder, VQ codebook, GAN/VGG losses
├── muse_maskgit_SSST.py     # Transformer, MaskGit, MaskGitTransformer, TokenCritic, Muse
└── trainer.py               # VQGanVAETrainer, MaskGitTrainer (accelerate + EMA)
```

## Data Format

Trans2Spatial expects a data directory (e.g. `../data/Dataset4/`) containing three files:

| File | Format | Shape | Description |
|------|--------|-------|-------------|
| `scRNA_count_sorted.csv` | CSV, comma-separated, header + index | `(num_genes, num_cells)` | scRNA-seq count matrix; rows are genes, columns are cells. `num_cells` feeds the VQ codebook size. |
| `Insitu_count.txt` | TSV, header | `(num_spots, num_genes)` | In-situ spatial expression counts per spot. |
| `Locations.txt` | TSV | `(num_spots, 2)` | `(x, y)` tissue coordinates per spot, scaled onto the image grid. |

`dataloader.get_num_cells(data_dir)` reads the scRNA matrix and returns its column count, which is used as the VQ codebook size so that each token maps to a cell.

## Installation

### Requirements

- Python ≥ 3.8
- Java 8 or 11 (required by PySpark, used for streaming large count files)

### Python dependencies

```bash
pip install torch torchvision
pip install accelerate ema-pytorch vector-quantize-pytorch
pip install einops beartype tqdm
pip install pyspark pandas numpy pillow
```

### Hardware

A CUDA GPU is strongly recommended. The default configuration (`image_size=512`, `seq_len=128**2`) is memory-intensive; adjust `batch_size` and `grad_accum_every` to fit your device. The code sets `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` to mitigate fragmentation.

## Usage

The full workflow is driven by `main.py`, organized as four sequential steps. Uncomment the step you want to run (Steps 1–3 are mutually exclusive training runs; Step 4 is inference). Below is the canonical configuration, mirroring `main.py`.

### Step 1 — Train the VQGAN-VAE

```python
from vqgan_vae import VQGanVAE
from trainer import VQGanVAETrainer

# 14249 is num_cells for the example dataset; set to your get_num_cells() value
vae = VQGanVAE(
    dim=256,
    vq_codebook_size=14249,   # = num_cells
)

trainer = VQGanVAETrainer(
    vae=vae,
    image_size=512,
    folder='../data/Dataset4',
    batch_size=4,
    grad_accum_every=8,
    num_train_steps=10000,
).cuda()

trainer.train()
```

### Step 2 — Train the base MaskGit

```python
from vqgan_vae import VQGanVAE
from muse_maskgit_SSST import MaskGitTransformer
from trainer import MaskGitTrainer
from dataloader import get_num_cells

vae = VQGanVAE(dim=256, vq_codebook_size=14249)
vae.load('results_vqsize_numcells/vae.8000.pt')

data_dir = '../data/Dataset4/'
num_cells = get_num_cells(data_dir)

transformer = MaskGitTransformer(
    num_tokens=num_cells,        # must equal the VQ codebook size
    seq_len=128 ** 2,            # must equal fmap_size ** 2 of the VAE
    dim=512,
    depth=8,
    dim_head=64,
    heads=8,
    ff_mult=4,
)

maskgit_trainer = MaskGitTrainer(
    vae=vae,
    transformer=transformer,
    folder=data_dir,
    num_train_steps=10000,
    batch_size=4,
    image_size=256,
    cond_drop_prob=0.25,
).cuda()

maskgit_trainer.train()
```

### Step 3 — Train the super-resolution MaskGit

```python
vae = VQGanVAE(dim=256, vq_codebook_size=14249)
vae.load('results_vqsize_numcells/vae.8000.pt')

data_dir = '../data/Dataset4/'
num_cells = get_num_cells(data_dir)

transformer = MaskGitTransformer(
    num_tokens=num_cells,
    seq_len=128 ** 2,
    dim=512,
    depth=2,
    dim_head=64,
    heads=8,
    ff_mult=4,
)

superres_trainer = MaskGitTrainer(
    vae=vae,
    transformer=transformer,
    folder=data_dir,
    is_superres=True,
    num_train_steps=10000,
    batch_size=4,
    image_size=512,
    cond_image_size=256,         # low-res conditioning size
    cond_drop_prob=0.25,
    results_folder='./Superres_maskgit_result',
).cuda()

superres_trainer.train()
```

### Step 4 — Generate spatial images with Muse

```python
from muse_maskgit_SSST import MaskGit, Muse

# Initialize base_maskgit and superres_maskgit with the same configuration
# used during training (see MaskGitTrainer.__init__, trainer.py lines ~523–539),
# then load the trained checkpoints.
base_maskgit.load('./path/to/base.pt')
superres_maskgit.load('./path/to/superres.pt')

muse = Muse(
    base=base_maskgit,
    superres=superres_maskgit,
)

# In the original Muse, texts are text prompts; here each entry should be the
# scRNA-seq count vector for a gene, following the same data format used in
# training. See the Notes & Caveats section.
images = muse([
    'a whale breaching from afar',          # placeholder; replace with scRNA inputs
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles',
    'waking up to a psychedelic landscape',
])

images  # generated spatial images
```

To resume training from a checkpoint, use `MaskGitTrainer.retrain(model_dir)`.

## Configuration & Key Hyperparameters

### VQGAN-VAE (`VQGanVAE`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | — | Base channel dimension of the encoder/decoder (e.g. `256`). |
| `channels` | `1` | Image channels. Spatial maps are single-channel; 3-channel mode just repeats the channel and requires editing `VQGanVAE`. |
| `layers` | `4` | Number of downsampling stages; determines `dim_divisor = 2 ** layers`. |
| `vq_codebook_size` | `512` | Number of VQ codes. **Set to `num_cells`.** |
| `vq_decay` | `0.8` | EMA decay for codebook updates. |
| `use_vgg_and_gan` | `False` | Enable discriminator + VGG perceptual loss (typically off for grayscale). |

### MaskGit Transformer (`MaskGitTransformer`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_tokens` | — | Must equal the VQ codebook size (`num_cells`). |
| `seq_len` | — | Must equal `fmap_size ** 2` of the VAE (e.g. `128 ** 2` for `image_size=512`, `layers=4`). |
| `dim` | — | Transformer model dimension (e.g. `512`). |
| `depth` | — | Number of transformer blocks (base: `8`, super-res: `2` in the example). |
| `dim_head` / `heads` | `64` / `8` | Attention configuration. |
| `ff_mult` | `4` | Feed-forward expansion factor. |
| `gene_embed_dim` | `18` | Dimension of the scRNA-seq cell embedding. |

### MaskGit (`MaskGit`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | — | Output spatial image size. |
| `cond_image_size` | `None` | Low-res conditioning size; required for super-resolution. |
| `cond_drop_prob` | `0.5` | Probability of dropping the condition for classifier-free guidance. |
| `self_cond_prob` | `0.9` | Probability of using self-conditioning during training. |
| `noise_schedule` | `cosine_schedule` | Masking schedule over timesteps. |
| `timesteps` | `18` | Number of iterative masking steps at inference (per the MaskGit paper). |
| `cond_scale` | `3` | Classifier-free guidance scale. |

### Trainers

Both trainers (`VQGanVAETrainer`, `MaskGitTrainer`) use HuggingFace `accelerate` for distributed/mixed-precision training and `ema_pytorch` for exponential moving averages. Common knobs: `lr` (`3e-4`), `grad_accum_every`, `max_grad_norm`, `save_model_every`, `save_results_every`, `valid_frac` (`0.05`), `results_folder`.

## Module Reference

### `dataloader.py`

- `scale_coordinates(coords, target_size)` — Linearly maps `(x, y)` coordinates into `[0, target_size-1]`.
- `get_num_cells(data_dir)` — Returns the number of cells (columns) in `scRNA_count_sorted.csv`.
- `scRNAseqSTDataset(Dataset)` — Loads the three data files via Spark, builds the `(H, W, num_genes)` spatial image tensor by placing in-situ counts at scaled coordinates, and pairs each gene's spatial slice with its scRNA-seq count vector. `__getitem__` returns `(scRNA_count[1D], spatial_image[1, H, W])`.

### `scRNAseqProcess.py`

- `gene_seq(insitu_count_file)` — Reads an in-situ count TSV with Spark and returns `(gene_to_idx, idx_to_gene)` dictionaries.

### `scRNASeqEmbed.py`

- `scRNASeqEmbedding(nn.Module)` — Embeds cell indices via a Xavier-initialized `nn.Embedding(num_cells, embedding_dim)` and concatenates them with the scRNA-seq count values, producing a `(num_genes, embedding_dim+1, num_cells)` tensor used as cross-attention context. `test_scRNASeqEmbedding()` provides a usage sanity check.

### `vqgan_vae.py`

- **Building blocks:** `LayerNormChan`, `ResBlock`, `GLUResBlock`, `ResnetEncDec` (ResNet encoder/decoder), `Discriminator` (PatchGAN-style).
- **Losses:** `hinge_discr_loss` / `hinge_gen_loss`, `bce_discr_loss` / `bce_gen_loss`, `gradient_penalty`.
- `VQGanVAE` — Main model. `encode` returns `(fmap, indices, commit_loss)`; `decode` / `decode_from_ids` reconstruct from codes or token ids. `forward` computes autoencoder or discriminator loss depending on flags. Includes `save`/`load` and `copy_for_eval` (strips VGG/discriminator for inference).

### `muse_maskgit_SSST.py`

- **Transformer primitives:** `LayerNorm`, `GEGLU`, `FeedForward`, `Attention` (with cross-attention + null-key support), `TransformerBlocks`, `Transformer`.
- `Transformer.forward` accepts scRNA-seq conditioning via `scRNA_count` or precomputed `gene_embeds`, applies classifier-free guidance dropout, optional self-conditioning, and optional low-res `conditioning_gene_ids`.
- `SelfCritic`, `MaskGitTransformer` (adds a mask token), `TokenCritic` (`dim_out=1`).
- **Sampling helpers:** `gumbel_sample`, `top_k`, `cosine_schedule`, `prob_mask_like`.
- `MaskGit` — Wraps the VAE + transformer; `forward` computes the masked-prediction training loss (cross-entropy + optional token-critic BCE); `generate` performs iterative masked decoding with annealed temperature and confidence-based unmasking.
- `Muse` — Combines `base` and `superres` MaskGits; `forward` generates a low-res map then super-resolves it.

### `trainer.py`

- `VQGanVAETrainer` — Trains the VQGAN-VAE with optional discriminator, EMA, periodic reconstruction grids, and checkpointing.
- `MaskGitTrainer` — Trains a base or super-resolution (`is_superres=True`) MaskGit with periodic validation loss, checkpointing, and a `retrain(model_dir)` entry point for resuming.
- `ImageDataset` — Internal dataset used by `VQGanVAETrainer` that yields spatial image slices (no scRNA pairing).

### `main.py`

End-to-end driver with four labeled, comment-toggleable steps (train VAE → train base MaskGit → train super-res MaskGit → Muse inference). See [Usage](#usage).

## Notes & Caveats

- **`main.py` Step 4 is a scaffold.** The `Muse` call still uses placeholder text prompts. The `Muse` / `MaskGit.generate` interface needs to be adapted to accept scRNA-seq count vectors in the same format used during training (the `texts` argument currently flows into `scRNASeqEmbedding`); `generate` also references `self.transformer.encode_text` and `conditioning_token_ids` names that are not fully wired up in `muse_maskgit_SSST.py`. Treat Step 4 as a starting point to complete.
- **Codebook size = `num_cells`.** The VQ codebook and the transformer token vocabulary are both sized to the number of cells, so the dataset's `num_cells` must be consistent across all four steps.
- **`seq_len` must match the VAE feature-map size.** For `image_size=512` and `layers=4`, the encoded feature map is `128×128`, so `seq_len = 128**2`. Mismatched values will raise shape errors.
- **Grayscale by default.** `channels=1`; the VGG/perceptual + GAN path is disabled. Enabling 3 channels requires modifying `VQGanVAE` and the dataset's `unsqueeze` logic.
- **Spark for large files.** The dataset loader and `scRNAseqProcess.py` use PySpark to read tab/comma-separated count files; ensure a Java runtime is available.
- **Single-channel `num_cells=14249`** in the examples reflects the bundled example dataset — replace with your own value from `get_num_cells()`.
- **`.DS_Store` / `._*` AppleDouble files** may appear on macOS external volumes; they are not part of the project and can be ignored or removed with `find . -name '._*' -delete`.

## Acknowledgements

This project adapts the Muse / MaskGit implementation by [lucidrains](https://github.com/lucidrains/muse-maskgit-pytorch). It builds on:

- **Muse** — Chang et al., *Muse: Text-To-Image Generation via Masked Generative Transformers*.
- **MaskGit** — Chang et al., *MaskGit: Masked Generative Image Transformer*.
- **VQGAN** — Esser, Rombach & Ommer, *Taming Transformers for High-Resolution Image Synthesis*.
- [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch), [accelerate](https://github.com/huggingface/accelerate), [ema-pytorch](https://github.com/lucidrains/ema-pytorch).

## License

This project is provided for research purposes. The adapted Muse/MaskGit code follows the terms of its upstream repository (MIT). Please respect the licenses of all dependencies and datasets used.
