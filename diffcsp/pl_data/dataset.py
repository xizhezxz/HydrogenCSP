import os

import hydra
import omegaconf
import torch
from omegaconf import ValueNode
from torch.utils.data import Dataset
from torch_geometric.data import Data
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from diffcsp.common.data_utils import (
    crystal_to_diffusion_data,
    get_symmetry_info,
    get_symmetry_info_original_cell,
    preprocess,
    preprocess_tensors,
)
from diffcsp.common.utils import PROJECT_ROOT


class CrystDataset(Dataset):
    """Dataset containing only fields consumed by CSPDiffusion."""

    def __init__(self, name: ValueNode, path: ValueNode,
                 niggli: ValueNode, primitive: ValueNode,
                 preprocess_workers: ValueNode, save_path: ValueNode,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.niggli = niggli
        self.primitive = primitive
        self.preprocess(save_path, preprocess_workers)

    def preprocess(self, save_path, preprocess_workers):
        required = {
            'frac_coords', 'atom_types', 'lengths', 'angles', 'num_atoms'
        }
        if os.path.exists(save_path):
            # Project caches contain NumPy/Python objects, not model weights.
            self.cached_data = torch.load(save_path, weights_only=False)
            if all(required.issubset(item) for item in self.cached_data):
                return

        cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
        )
        torch.save(cached_data, save_path)
        self.cached_data = cached_data

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, index):
        item = self.cached_data[index]
        return Data(
            frac_coords=torch.as_tensor(item['frac_coords'], dtype=torch.float),
            atom_types=torch.as_tensor(item['atom_types'], dtype=torch.long),
            lengths=torch.as_tensor(item['lengths'], dtype=torch.float).view(1, -1),
            angles=torch.as_tensor(item['angles'], dtype=torch.float).view(1, -1),
            num_atoms=int(item['num_atoms']),
            num_nodes=int(item['num_atoms']),
        )

    def __repr__(self):
        return f"CrystDataset(name={self.name!r}, path={self.path!r})"


def add_spacegroup_constraints(
        data, tolerance=0.01, preserve_cell=False):
    """Convert one crystal graph to a PyXtal Wyckoff-orbit graph.

    PyXtal may standardize the cell and reorder its sites.  The returned
    coordinates, atom types, and lattice therefore all come from the same
    standardized structure as the returned Wyckoff operations.
    """
    lengths = data.lengths.detach().cpu().reshape(-1).numpy()
    angles = data.angles.detach().cpu().reshape(-1).numpy()
    frac_coords = data.frac_coords.detach().cpu().numpy()
    atom_types = data.atom_types.detach().cpu().numpy()
    crystal = Structure(
        lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
        species=atom_types,
        coords=frac_coords,
        coords_are_cartesian=False,
    )
    symmetry_fn = (
        get_symmetry_info_original_cell
        if preserve_cell else get_symmetry_info
    )
    crystal, symmetry = symmetry_fn(crystal, tol=tolerance)
    crystal_data = crystal_to_diffusion_data(crystal)

    ops = torch.as_tensor(symmetry['wyckoff_ops'], dtype=torch.float)
    if ops.shape != (crystal_data['num_atoms'], 4, 4):
        raise ValueError(
            "Wyckoff preprocessing must return one 4x4 operation per site"
        )

    return Data(
        frac_coords=torch.as_tensor(
            crystal_data['frac_coords'], dtype=torch.float),
        atom_types=torch.as_tensor(
            crystal_data['atom_types'], dtype=torch.long),
        lengths=torch.as_tensor(
            crystal_data['lengths'], dtype=torch.float).view(1, -1),
        angles=torch.as_tensor(
            crystal_data['angles'], dtype=torch.float).view(1, -1),
        num_atoms=int(crystal_data['num_atoms']),
        num_nodes=int(crystal_data['num_atoms']),
        spacegroup=torch.tensor(
            [symmetry['spacegroup']], dtype=torch.long),
        ops=ops,
        anchor_index=torch.as_tensor(
            symmetry['anchors'], dtype=torch.long),
        ops_inv=torch.linalg.pinv(ops[:, :3, :3]),
    )


class SpaceGroupConstraintDataset(Dataset):
    """Lazily add PyXtal Wyckoff constraints to another crystal dataset."""

    def __init__(self, dataset, tolerance=0.01, preserve_cell=False):
        super().__init__()
        self.dataset = dataset
        self.tolerance = tolerance
        self.preserve_cell = preserve_cell

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return add_spacegroup_constraints(
            self.dataset[index], tolerance=self.tolerance,
            preserve_cell=self.preserve_cell)

    def __repr__(self):
        return (
            f"SpaceGroupConstraintDataset(dataset={self.dataset!r}, "
            f"tolerance={self.tolerance}, preserve_cell={self.preserve_cell})"
        )


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, preprocess_workers, **kwargs):
        super().__init__()
        self.cached_data = preprocess_tensors(
            crystal_array_list, num_workers=preprocess_workers)

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, index):
        item = self.cached_data[index]
        return Data(
            frac_coords=torch.as_tensor(item['frac_coords'], dtype=torch.float),
            atom_types=torch.as_tensor(item['atom_types'], dtype=torch.long),
            lengths=torch.as_tensor(item['lengths'], dtype=torch.float).view(1, -1),
            angles=torch.as_tensor(item['angles'], dtype=torch.float).view(1, -1),
            num_atoms=int(item['num_atoms']),
            num_nodes=int(item['num_atoms']),
        )

    def __repr__(self):
        return f"TensorCrystDataset(len={len(self.cached_data)})"


@hydra.main(
    version_base="1.3",
    config_path=str(PROJECT_ROOT / "conf"),
    config_name="default",
)
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch

    dataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False)
    return Batch.from_data_list([dataset[i] for i in range(len(dataset))])


if __name__ == "__main__":
    main()
