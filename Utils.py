
import scanpy as sc
import anndata as ad
import pathlib
import urllib.request
import urllib.error
import os
from pathlib import Path as PathLike
import squidpy as sq
from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# *******************************************
# data fucntions
# *******************************************

def _download(url: str, path: pathlib.Path) -> None:  # pragma: no cover
    """Download file from url to path."""
    try:
        # Ensure the parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading data from {url} to {path}...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download from {url}. Error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during download: {e}") from e
def _load_dataset_from_url(
    path: PathLike,
    file_type: str,
    backup_url: str,
    expected_shape: tuple[int, int],
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:  # pragma: no cover
    """Load a dataset from a URL or a local path."""
    path_obj = pathlib.Path(os.path.expanduser(path))
    if force_download or not path_obj.is_file():
        _download(backup_url, path_obj)
    else:
        print(f"Found local copy of data at {path_obj}.")
    if file_type.lower() == "h5ad":
        adata = sc.read_h5ad(path_obj, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}. Only 'h5ad' is supported here.")
    if adata.shape != expected_shape:
        print(
            f"Warning: AnnData shape is {adata.shape}, but expected {expected_shape}. "
        )
    return adata

def mosta(
    path: PathLike = "~/.cache/moscot/mosta.h5ad", # Default cache path
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:  # pragma: no cover
    """Preprocessed and extracted data as provided in :cite:`chen:22`.
    Includes embryo sections `E9.5`, `E2S1`, `E10.5`, `E2S1`, `E11.5`, `E1S2`.
    The :attr:`anndata.AnnData.X` is based on reprocessing of the counts data using
    :func:`scanpy.pp.normalize_total` and :func:`scanpy.pp.log1p`.
    Parameters
    ----------
    path
        Path where to save the file.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`scanpy.read_h5ad`.
    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(
        path,
        file_type="h5ad",
        backup_url="https://figshare.com/ndownloader/files/40569779",
        expected_shape=(54134, 2000),
        force_download=force_download,
        **kwargs,
    )

def process_data(data):
    # Filter genes
    sc.pp.filter_genes(data, min_counts=100)
    # Compute spatial neighbors
    sq.gr.spatial_neighbors(data, key_added='spatial')
    # Compute Moran's I
    sq.gr.spatial_autocorr(
        data,
        genes=data.var_names,  # calculate for all genes
        mode='moran',
        n_jobs=1  # Adjust the number of parallel jobs if needed
    )
    return len(data.var_names)  # Return the number of genes after filtering

# Function to prepare gene expression target
def prepare_gene_expression_target(adata_e11_5, gene_names, target_size=1000):
    """
    Convert spatial gene expression data to a target image for NCA training.
    """
    # Get spatial coordinates and gene expression
    spatial_coords = adata_e11_5.obsm['spatial']
    all_gene_expression = []
    for gene_name in gene_names:
        if isinstance(adata_e11_5.X, np.ndarray):
            gene_expression = adata_e11_5[:, gene_name].X.flatten()
        else:
            gene_expression = adata_e11_5[:, gene_name].X.toarray().flatten()
        all_gene_expression.append(gene_expression)
    all_gene_expression = np.vstack(all_gene_expression).T # Shape (n_obs, n_genes)
  
    # Normalize coordinates to [0, target_size-1]
    coords_norm = spatial_coords.copy()
    coords_norm[:, 0] = (coords_norm[:, 0] - coords_norm[:, 0].min()) / (coords_norm[:, 0].max() - coords_norm[:, 0].min()) * (target_size - 1)
    coords_norm[:, 1] = (coords_norm[:, 1] - coords_norm[:, 1].min()) / (coords_norm[:, 1].max() - coords_norm[:, 1].min()) * (target_size - 1)

    # Create target image
    target_img = np.zeros((target_size, target_size, len(gene_names)), dtype=np.float32)

    for i, (x, y) in enumerate(coords_norm.astype(int)):
        if 0 <= x < target_size and 0 <= y < target_size:
            # RGB channels represent gene expression intensity
            for c in range(len(gene_names)):
                target_img[y, x, c] = all_gene_expression[i, c]

    return target_img # Return both target_img and scaler


def create_nca_input(
    adata_e11_5,
    genes: list[str],
    hidden_channels: int = 100,
    target_size: int = 100,
    pad_to: Optional[int] = None,
) -> np.ndarray:
    """
    Uses prepare_gene_expression_target to create the RGB image, then adds
    alpha and latent channels, normalizes RGB, and (optionally) pads to square pad_to.
    Returns the final (H, W, C) numpy array for NCA training.
    """
    # ...existing code...
    target_rgb = prepare_gene_expression_target(adata_e11_5, genes, target_size)
    h, w = target_rgb.shape[:2]
    alpha = (target_rgb.sum(-1, keepdims=True) > 0).astype(np.float32)
    latent = np.zeros((h, w, hidden_channels), dtype=np.float32)
    target_img_np = np.concatenate([target_rgb, alpha, latent], axis=-1)
    for c in range(3):
        ch = target_img_np[..., c]
        target_img_np[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

    # ---- newly added padding block ----
    if pad_to is not None and pad_to > target_size:
        pad_h = pad_to - h
        pad_w = pad_to - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        target_img_np = np.pad(
            target_img_np,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    # ------------------------------------

    return target_img_np



# *******************************************
# Model
# *******************************************

class CAModel(nn.Module):
    def __init__(self, n_genes=10, hidden_channels=10, fire_rate=0.5, alive_threshold=0.1):
        super(CAModel, self).__init__()
        self.n_genes = n_genes
        self.hidden_channels = hidden_channels
        self.channel_n = n_genes + 1 + hidden_channels  # genes + alpha + hidden
        self.fire_rate = fire_rate
        self.alive_thr = alive_threshold
        
        # Define fixed perception kernels (identity, Sobel X, Sobel Y, Laplacian)
        identity_kernel_vals = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]], dtype=torch.float32)
        sobel_x_kernel_vals = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32) / 8.0
        sobel_y_kernel_vals = sobel_x_kernel_vals.T  # Transpose of Sobel X is Sobel Y
        laplacian_kernel_vals = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32)

        # Expand kernels for depthwise-like application for each channel
        k_identity = identity_kernel_vals.unsqueeze(0).unsqueeze(0).repeat(self.channel_n, 1, 1, 1)
        k_dx = sobel_x_kernel_vals.unsqueeze(0).unsqueeze(0).repeat(self.channel_n, 1, 1, 1)
        k_dy = sobel_y_kernel_vals.unsqueeze(0).unsqueeze(0).repeat(self.channel_n, 1, 1, 1)
        k_laplacian = laplacian_kernel_vals.unsqueeze(0).unsqueeze(0).repeat(self.channel_n, 1, 1, 1)

        # Concatenate to form the full perception kernel
        full_perception_kernel = torch.cat([k_identity, k_dx, k_dy, k_laplacian], dim=0)
        self.register_buffer('perception_kernel', full_perception_kernel)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(4*self.channel_n, 128, 1)  # Adjust input channels to account for Laplacian
        self.conv2 = nn.Conv2d(128, self.channel_n, 1, bias=False)
        nn.init.zeros_(self.conv2.weight)    

    def alive(self, x):
        # Alpha channel is at position n_genes (after all gene channels)
        alpha_idx = self.n_genes
        a = x[:, alpha_idx:alpha_idx+1]
        return F.max_pool2d((a > self.alive_thr).float(), 3, 1, 1)

    def forward(self, x, fire_rate=None, step_size=1.0):
        # Perceive the environment
        pre_alive = self.alive(x)
        y = self.perceive(x)
        
        # Apply ReLU after the first convolution, then the second convolution
        dx = self.conv2(F.relu(self.conv1(y))) * step_size
        
        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = (torch.rand_like(x[:, :1]) <= fire_rate).float()
        x = x + dx * update_mask  
        
        post_alive = self.alive(x)
        life_mask = (pre_alive * post_alive) 
        x = x * life_mask  # clear "dead" cells
        
        # Clamp gene expression channels and alpha to [0,1]
        gene_alpha_end = self.n_genes + 1
        x[:, :gene_alpha_end].clamp_(0.0, 1.0)
        # Clamp hidden channels to [-10, 10]
        x[:, gene_alpha_end:].clamp_(-10.0, 10.0)
        
        return x
    
    def perceive(self, x):
        # Apply perception kernels (identity, Sobel X, Sobel Y, Laplacian)
        y = F.conv2d(x, self.perception_kernel, stride=1, padding=1, groups=self.channel_n)
        return y
