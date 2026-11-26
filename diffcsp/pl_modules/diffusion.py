import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)

from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal

MAX_ATOMIC_NUM=100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                # Validation is configured every N epochs, so plateau
                # scheduling must follow the same cadence on Lightning 2.x.
                "frequency": self.hparams.logging.val_check_interval,
            },
        }


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, _recursive_=False)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        # ``keep_lattice`` is an explicit modelling choice.  Falling back to
        # the old loss-weight convention keeps existing checkpoints/configs
        # loadable.
        self.keep_lattice = getattr(
            self.hparams, "keep_lattice", self.hparams.cost_lattice < 1e-5
        )
        self.keep_coords = self.hparams.cost_coord < 1e-5
        if self.keep_lattice and hasattr(self.decoder, "lattice_out"):
            self.decoder.lattice_out.requires_grad_(False)

    def forward(self, batch):

        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_x = torch.randn_like(frac_coords)
        if self.keep_lattice:
            # The clean lattice is conditioning information, not a diffusion
            # variable, when only atomic positions are generated.
            rand_l = torch.zeros_like(lattices)
            input_lattice = lattices
        else:
            rand_l = torch.randn_like(lattices)
            input_lattice = (
                c0[:, None, None] * lattices
                + c1[:, None, None] * rand_l
            )
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.


        if self.keep_coords:
            input_frac_coords = frac_coords

        pred_l, pred_x = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        loss_lattice = (
            pred_l.new_zeros(())
            if self.keep_lattice
            else F.mse_loss(pred_l, rand_l)
        )
        loss_coord = F.mse_loss(pred_x, tar_x)

        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord
        }

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5):

        batch_size = batch.num_graphs

        if self.keep_lattice and not (
                hasattr(batch, "lengths") and hasattr(batch, "angles")):
            raise ValueError(
                "Fixed-lattice sampling requires batch.lengths and "
                "batch.angles as the lattice condition."
            )

        x_T = torch.rand([batch.num_nodes, 3], device=self.device)
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        else:
            l_T = torch.randn([batch_size, 3, 3], device=self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]


            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector

            rand_l = (
                torch.randn_like(l_T)
                if t > 1 and not self.keep_lattice
                else torch.zeros_like(l_T)
            )
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            # step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_05 = l_t

            # Predictor

            rand_l = (
                torch.randn_like(l_T)
                if t > 1 and not self.keep_lattice
                else torch.zeros_like(l_T)
            )
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_1 = (
                l_T
                if self.keep_lattice
                else c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l
            )


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack

    @torch.no_grad()
    def sample_spacegroup(
            self, batch, step_lr=1e-5, constrained_mask=None):
        """Sample coordinates while enforcing the input space-group orbits.

        The lattice, composition, and Wyckoff-orbit assignment are fixed.  A
        coordinate is sampled for each orbit representative and expanded to
        all symmetry-equivalent sites with ``batch.ops`` after initialization
        and after every predictor/corrector update.  Coordinate scores are
        projected to the representative frame, averaged within each orbit,
        and expanded back before they are applied.

        In addition to the fields required by :meth:`sample`, ``batch`` must
        provide:

        * ``spacegroup``: one international space-group number per crystal;
        * ``ops``: one 4x4 fractional affine operation per atomic site;
        * ``anchor_index``: the representative atom index for each site;
        * ``ops_inv`` (optional): inverse/pseudoinverse of each operation's
          3x3 rotational block.

        ``frame_index`` is optional and defaults to all atomic sites.  When it
        is supplied, it must still be a permutation covering every site.
        ``constrained_mask`` is an optional boolean mask with one value per
        site.  By default every site is constrained.  Sites set to False keep
        independent coordinates, coordinate scores, and sampling noise.
        """
        required = (
            "lengths", "angles", "atom_types", "num_atoms", "batch",
            "spacegroup", "ops", "anchor_index",
        )
        missing = [name for name in required if not hasattr(batch, name)]
        if missing:
            raise ValueError(
                "Space-group sampling requires batch fields: "
                + ", ".join(missing)
            )

        batch_size = batch.num_graphs
        num_nodes = batch.num_nodes
        spacegroup = batch.spacegroup.to(self.device).reshape(-1)
        if spacegroup.numel() != batch_size:
            raise ValueError(
                "batch.spacegroup must contain one value per crystal"
            )
        if torch.any((spacegroup < 1) | (spacegroup > 230)):
            raise ValueError("Space-group numbers must lie in [1, 230]")

        ops = batch.ops.to(self.device)
        anchor_index = batch.anchor_index.to(
            self.device, dtype=torch.long).reshape(-1)
        frame_index = getattr(
            batch,
            "frame_index",
            torch.arange(num_nodes, device=self.device),
        ).to(self.device, dtype=torch.long).reshape(-1)

        if ops.ndim != 3 or ops.shape[1:] != (4, 4):
            raise ValueError("batch.ops must have shape [num_nodes, 4, 4]")
        if not (
                ops.size(0) == anchor_index.numel()
                == frame_index.numel() == num_nodes):
            raise ValueError(
                "ops, anchor_index, and frame_index must contain one entry "
                "per atomic site"
            )
        if torch.any((anchor_index < 0) | (anchor_index >= num_nodes)):
            raise ValueError("batch.anchor_index contains an invalid atom index")
        if not torch.equal(
                torch.sort(frame_index).values,
                torch.arange(num_nodes, device=self.device)):
            raise ValueError(
                "batch.frame_index must be a permutation of all atom indices"
            )
        node_batch = batch.batch.to(self.device)
        if not torch.equal(
                node_batch[frame_index], node_batch[anchor_index]):
            raise ValueError(
                "Each symmetry-equivalent site and its anchor must belong "
                "to the same crystal"
            )
        atom_types = batch.atom_types.to(self.device)
        if not torch.equal(
                atom_types[frame_index], atom_types[anchor_index]):
            raise ValueError(
                "All sites in a symmetry orbit must have the anchor's "
                "atomic type"
            )

        if constrained_mask is None:
            constrained_mask = torch.ones(
                num_nodes, device=self.device, dtype=torch.bool)
        else:
            constrained_mask = constrained_mask.to(
                self.device, dtype=torch.bool).reshape(-1)
            if constrained_mask.numel() != num_nodes:
                raise ValueError(
                    "constrained_mask must contain one value per atomic site"
                )
        frame_is_constrained = constrained_mask[frame_index]
        if not torch.equal(
                frame_is_constrained, constrained_mask[anchor_index]):
            raise ValueError(
                "constrained_mask must select either every site or no site "
                "from each symmetry orbit"
            )
        constrained_frames = frame_index[frame_is_constrained]
        constrained_anchors = anchor_index[frame_is_constrained]

        rotations = ops[:, :3, :3]
        if hasattr(batch, "ops_inv"):
            ops_inv = batch.ops_inv.to(self.device)
            if ops_inv.shape != rotations.shape:
                raise ValueError("batch.ops_inv must have shape [num_nodes, 3, 3]")
        else:
            ops_inv = torch.linalg.pinv(rotations)

        def expand_anchor_coords(coords):
            if constrained_frames.numel() == 0:
                return coords
            result = coords.clone()
            constrained_ops = ops[frame_is_constrained]
            unique_anchors = torch.unique(constrained_anchors)
            anchor_rows = torch.stack([
                torch.nonzero(
                    constrained_frames.eq(anchor), as_tuple=False
                ).flatten()[0]
                for anchor in unique_anchors
            ])
            anchor_ops = constrained_ops[anchor_rows]
            anchor_coords = result[unique_anchors]

            # Affine projectors for special Wyckoff positions need a
            # consistent unwrapped representative.  Search neighboring
            # periodic images and choose the projected image closest to the
            # current wrapped anchor.  Applying a fractional projector after
            # an unconditional ``% 1`` can otherwise jump by half a cell.
            shifts = torch.cartesian_prod(*[
                torch.tensor(
                    [0.0, -1.0, 1.0],
                    device=coords.device,
                    dtype=coords.dtype,
                )
                for _ in range(3)
            ])
            candidates = anchor_coords[:, None, :] + shifts[None, :, :]
            homogeneous_candidates = torch.cat([
                candidates,
                torch.ones(
                    (*candidates.shape[:2], 1),
                    device=coords.device,
                    dtype=coords.dtype,
                ),
            ], dim=-1)
            projected = torch.einsum(
                "aij,asj->asi", anchor_ops, homogeneous_candidates
            )[:, :, :3]
            periodic_error = torch.remainder(
                projected - anchor_coords[:, None, :] + 0.5, 1.0
            ) - 0.5
            best_images = periodic_error.square().sum(dim=-1).argmin(dim=-1)
            representatives = projected[
                torch.arange(unique_anchors.numel(), device=coords.device),
                best_images,
            ]
            representative_by_node = torch.zeros_like(result)
            representative_by_node[unique_anchors] = representatives

            homogeneous = torch.cat([
                representative_by_node[constrained_anchors],
                torch.ones(
                    (constrained_frames.numel(), 1),
                    device=coords.device,
                    dtype=coords.dtype,
                ),
            ], dim=-1)
            expanded = torch.bmm(
                constrained_ops, homogeneous.unsqueeze(-1))
            result[constrained_frames] = expanded[:, :3, 0] % 1.0
            return result

        def project_orbit_vectors(vectors):
            if constrained_frames.numel() == 0:
                return vectors
            representative_frame = torch.bmm(
                ops_inv[frame_is_constrained],
                vectors[constrained_frames].unsqueeze(-1),
            ).squeeze(-1)
            representative_mean = scatter(
                representative_frame,
                constrained_anchors,
                dim=0,
                dim_size=num_nodes,
                reduce="mean",
            )
            expanded = torch.bmm(
                rotations[frame_is_constrained],
                representative_mean[constrained_anchors].unsqueeze(-1),
            ).squeeze(-1)
            result = vectors.clone()
            result[constrained_frames] = expanded
            return result

        def symmetric_noise(reference, add_noise):
            if not add_noise:
                return torch.zeros_like(reference)
            # Unconstrained sites receive ordinary independent noise.
            noise = torch.randn_like(reference)
            if constrained_frames.numel() == 0:
                return noise
            anchor_noise = torch.zeros_like(reference)
            unique_anchors = torch.unique(constrained_anchors)
            anchor_noise[unique_anchors] = torch.randn(
                (unique_anchors.numel(), 3),
                device=reference.device,
                dtype=reference.dtype,
            )
            noise[constrained_frames] = torch.bmm(
                rotations[frame_is_constrained],
                anchor_noise[constrained_anchors].unsqueeze(-1),
            ).squeeze(-1)
            return noise

        l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        x_T = torch.rand((num_nodes, 3), device=self.device)
        if self.keep_coords:
            if not hasattr(batch, "frac_coords"):
                raise ValueError(
                    "Keeping coordinates requires batch.frac_coords"
                )
            x_T = batch.frac_coords.to(self.device)
        x_T = expand_anchor_coords(x_T)

        time_start = self.beta_scheduler.timesteps
        traj = {time_start: {
            "num_atoms": batch.num_atoms,
            "atom_types": batch.atom_types,
            "frac_coords": x_T,
            "lattices": l_T,
            "spacegroup": spacegroup,
        }}

        for t in tqdm(range(time_start, 0, -1)):
            times = torch.full(
                (batch_size,), t, device=self.device, dtype=torch.long
            )
            time_emb = self.time_embedding(times)
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            x_t = expand_anchor_coords(traj[t]["frac_coords"])

            # Corrector: update one shared degree of freedom per orbit.
            rand_x = symmetric_noise(x_T, add_noise=t > 1)
            step_size = step_lr * (
                sigma_x / self.sigma_scheduler.sigma_begin
            ) ** 2
            std_x = torch.sqrt(2 * step_size)
            _, pred_x = self.decoder(
                time_emb, batch.atom_types, x_t, l_T,
                batch.num_atoms, batch.batch,
            )
            pred_x = project_orbit_vectors(
                pred_x * torch.sqrt(sigma_norm)
            )
            x_half = (
                x_t
                if self.keep_coords
                else x_t - step_size * pred_x + std_x * rand_x
            )
            x_half = expand_anchor_coords(x_half)

            # Predictor: repeat the same symmetry projection before updating.
            rand_x = symmetric_noise(x_T, add_noise=t > 1)
            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = sigma_x ** 2 - adjacent_sigma_x ** 2
            std_x = torch.sqrt(
                adjacent_sigma_x ** 2 * step_size / sigma_x ** 2
            )
            _, pred_x = self.decoder(
                time_emb, batch.atom_types, x_half, l_T,
                batch.num_atoms, batch.batch,
            )
            pred_x = project_orbit_vectors(
                pred_x * torch.sqrt(sigma_norm)
            )
            x_next = (
                x_t
                if self.keep_coords
                else x_half - step_size * pred_x + std_x * rand_x
            )
            x_next = expand_anchor_coords(x_next)

            traj[t - 1] = {
                "num_atoms": batch.num_atoms,
                "atom_types": batch.atom_types,
                "frac_coords": x_next,
                "lattices": l_T,
                "spacegroup": spacegroup,
            }

        traj_stack = {
            "num_atoms": batch.num_atoms,
            "atom_types": batch.atom_types,
            "spacegroup": spacegroup,
            "all_frac_coords": torch.stack([
                traj[i]["frac_coords"]
                for i in range(time_start, -1, -1)
            ]),
            "all_lattices": torch.stack([
                traj[i]["lattices"]
                for i in range(time_start, -1, -1)
            ]),
        }

        return traj[0], traj_stack

    @torch.no_grad()
    def sample_spacegroup_hydrogen(self, batch, step_lr=1e-5):
        """Sample all coordinates, constraining only the non-H atoms.

        This is identical to :meth:`sample_spacegroup` except that hydrogen
        atoms (atomic number 1) retain independent coordinates, decoder
        scores, and predictor/corrector noise.  The lattice remains fixed.
        """
        if not hasattr(batch, "atom_types"):
            raise ValueError(
                "Space-group hydrogen sampling requires batch.atom_types"
            )
        non_hydrogen_mask = batch.atom_types.ne(1)
        return self.sample_spacegroup(
            batch,
            step_lr=step_lr,
            constrained_mask=non_hydrogen_mask,
        )

    @torch.no_grad()
    def sample_hydrogen(
            self, batch, step_lr=1e-5, return_trajectory=False):
        """Reconstruct H positions while keeping lattice and framework fixed.

        Hydrogen atoms (atomic number 1) are initialized uniformly in
        fractional space and updated by the predictor-corrector sampler.
        Every non-H coordinate and every input lattice is re-imposed before
        and after each denoising update, so they cannot drift numerically.

        Args:
            batch: A PyG batch containing ``frac_coords``, ``atom_types``,
                ``lengths``, ``angles``, ``num_atoms`` and ``batch``.
            step_lr: Corrector step size used by the standard CSP sampler.
            return_trajectory: Return all coordinate/lattice frames when True.
                The default avoids retaining 1001 frames during bulk sampling.

        Returns:
            ``(output, trajectory)``. ``trajectory`` is None unless requested.
        """
        required = ("frac_coords", "atom_types", "lengths", "angles",
                    "num_atoms", "batch")
        missing = [name for name in required if not hasattr(batch, name)]
        if missing:
            raise ValueError(
                "Hydrogen reconstruction requires batch fields: "
                + ", ".join(missing)
            )

        batch_size = batch.num_graphs
        fixed_frac_coords = batch.frac_coords.to(self.device) % 1.0
        hydrogen_mask = batch.atom_types.eq(1)
        hydrogen_counts = torch.bincount(
            batch.batch[hydrogen_mask], minlength=batch_size)
        missing_hydrogen = torch.nonzero(
            hydrogen_counts.eq(0), as_tuple=False).flatten()
        if missing_hydrogen.numel():
            missing_indices = missing_hydrogen.detach().cpu().tolist()
            raise ValueError(
                "sample_hydrogen requires at least one hydrogen atom in "
                "every structure; no hydrogen was found at batch indices: "
                f"{missing_indices}"
            )
        movable_mask = hydrogen_mask[:, None]
        fixed_lattice = lattice_params_to_matrix_torch(
            batch.lengths, batch.angles)

        random_frac_coords = torch.rand_like(fixed_frac_coords)
        x_t = torch.where(
            movable_mask, random_frac_coords, fixed_frac_coords)

        coord_trajectory = [x_t.clone()] if return_trajectory else None
        time_start = self.beta_scheduler.timesteps

        for t in tqdm(range(time_start, 0, -1)):
            times = torch.full(
                (batch_size,), t, device=self.device, dtype=torch.long)
            time_emb = self.time_embedding(times)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            # Keep the known framework and cell as exact conditioning values.
            x_t = torch.where(movable_mask, x_t, fixed_frac_coords)

            # Corrector: stochastic updates are restricted to hydrogen atoms.
            rand_x = torch.zeros_like(x_t)
            if t > 1:
                rand_x[hydrogen_mask] = torch.randn_like(
                    x_t[hydrogen_mask])
            step_size = step_lr * (
                sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            _, pred_x = self.decoder(
                time_emb, batch.atom_types, x_t, fixed_lattice,
                batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)
            proposed_x = x_t - step_size * pred_x + std_x * rand_x
            x_half = torch.where(
                movable_mask, proposed_x % 1.0, fixed_frac_coords)

            # Predictor: lattice noise and framework-coordinate updates are
            # deliberately omitted, independent of the model training mode.
            rand_x.zero_()
            if t > 1:
                rand_x[hydrogen_mask] = torch.randn_like(
                    x_t[hydrogen_mask])
            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = sigma_x ** 2 - adjacent_sigma_x ** 2
            std_x = torch.sqrt(
                adjacent_sigma_x ** 2 * step_size / sigma_x ** 2)

            _, pred_x = self.decoder(
                time_emb, batch.atom_types, x_half, fixed_lattice,
                batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)
            proposed_x = x_half - step_size * pred_x + std_x * rand_x
            x_t = torch.where(
                movable_mask, proposed_x % 1.0, fixed_frac_coords)

            if return_trajectory:
                coord_trajectory.append(x_t.clone())

        output = {
            'num_atoms': batch.num_atoms,
            'atom_types': batch.atom_types,
            'frac_coords': x_t,
            'lattices': fixed_lattice,
            'hydrogen_mask': hydrogen_mask,
        }

        trajectory = None
        if return_trajectory:
            all_frac_coords = torch.stack(coord_trajectory)
            trajectory = {
                'num_atoms': batch.num_atoms,
                'atom_types': batch.atom_types,
                'hydrogen_mask': hydrogen_mask,
                'all_frac_coords': all_frac_coords,
                'all_lattices': fixed_lattice.unsqueeze(0).expand(
                    all_frac_coords.size(0), -1, -1, -1),
            }

        return output, trajectory



    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
            batch_size=batch.num_graphs,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord
        }

        return log_dict, loss

    
