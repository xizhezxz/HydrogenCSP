"""Random non-H Wyckoff allocation without reference orbit assignments.

Space-group setting and input/standard cell transform are given conditions.
No reference equivalent_atoms, Wyckoff labels, or site coordinates are used
to select occupations. H sites retain identity operations and independent noise.
"""
from collections import Counter
from itertools import product

import numpy as np
import torch
from pyxtal.symmetry import Group
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_geometric.data import Data


def key_op(op):
    translation = np.remainder(op[:3, 3], 1)
    translation[np.isclose(translation, 1, atol=1e-7)] = 0
    return tuple(np.round(op[:3, :3].ravel(), 7)) + tuple(np.round(translation, 7))


def catalog_from_reference(data, tolerance=0.01):
    """Use only SG setting, cell transform and element counts from reference."""
    lattice = Lattice.from_parameters(*data.lengths.reshape(-1).tolist(),
                                      *data.angles.reshape(-1).tolist())
    crystal = Structure(lattice, data.atom_types.tolist(), data.frac_coords.numpy())
    analyzer = SpacegroupAnalyzer(crystal, symprec=tolerance)
    dataset = analyzer.get_symmetry_dataset()
    if dataset is None:
        raise ValueError('Space group could not be identified')
    # spglib convention: x_standard = P @ x_input + p (mod standard lattice).
    pmat = np.asarray(dataset.transformation_matrix)
    origin = np.asarray(dataset.origin_shift)
    inverse = np.linalg.inv(pmat)
    ratio = abs(np.linalg.det(pmat))
    group = Group(int(dataset.hall_number), use_hall=True)
    catalog = []
    for wp in group.Wyckoff_positions:
        multiplicity = wp.multiplicity * ratio
        if not np.isclose(multiplicity, round(multiplicity), atol=1e-5):
            continue
        expected = int(round(multiplicity))
        unique = {}
        # Standard-cell integer translations supply cosets if input is a supercell.
        for radius in (0, 1, 2, 3):
            for shift in product(range(-radius, radius+1), repeat=3):
                for sym in wp.ops:
                    op = np.eye(4)
                    op[:3, :3] = inverse @ sym.rotation_matrix @ pmat
                    op[:3, 3] = inverse @ (sym.rotation_matrix @ origin + sym.translation_vector + shift - origin)
                    unique.setdefault(key_op(op), op)
            if len(unique) >= expected:
                break
        if len(unique) != expected:
            raise ValueError(f'Wrong transformed multiplicity: SG {group.number}, {wp}, {len(unique)} != {expected}')
        raw_ops=np.stack(list(unique.values()))
        first=raw_ops[0]
        first_pinv=np.linalg.pinv(first[:3,:3])
        normalized=raw_ops.copy()
        for k,op in enumerate(raw_ops):
            normalized[k,:3,:3]=op[:3,:3] @ first_pinv
            normalized[k,:3,3]=op[:3,3]-normalized[k,:3,:3] @ first[:3,3]
        catalog.append({'letter':wp.letter,'standard_multiplicity':wp.multiplicity,
                        'multiplicity':expected,'dof':int(wp.get_dof()),
                        'ops':normalized})
    # Check that the transformed general-position operations equal the input SG.
    original_ops=[]
    for rotation,translation in zip(dataset.rotations,dataset.translations):
        sym=np.eye(4);sym[:3,:3]=rotation;sym[:3,3]=translation
        original_ops.append(sym)
    general = max(catalog, key=lambda x:x['multiplicity'])
    # A non-symmetry-adapted supercell can hide point operations: only integer
    # fractional rotations preserve its translation lattice. The full standard
    # Wyckoff orbit is still expanded, including standard-cell translation cosets.
    compatible_ops=[op for op in general['ops']
                    if np.allclose(op[:3,:3],np.rint(op[:3,:3]),atol=2e-5)]
    def operation_distance(a,b):
        shift=a[:3,3]-b[:3,3];shift-=np.round(shift)
        if not np.allclose(a[:3,:3],b[:3,:3],atol=2e-5):return np.inf
        return np.linalg.norm(shift @ lattice.matrix)
    operation_error=max(min(operation_distance(op,ref) for ref in original_ops) for op in compatible_ops)
    if len(compatible_ops)!=len(original_ops) or operation_error>2*tolerance:
        raise ValueError(f'Wyckoff setting mismatch: SG {group.number}, Hall {dataset.hall_number}; table_ops={len(general["ops"])}, input_ops={len(original_ops)}, max_error_A={operation_error}, P={pmat.tolist()}, origin={origin.tolist()}')
    return {'spacegroup':int(dataset.number),'hall_number':int(dataset.hall_number),
            'transformation_matrix':pmat.tolist(),'origin_shift':origin.tolist(),
            'volume_ratio_input_over_standard':ratio,'positions':catalog,
            'max_reference_operation_deviation_ang':operation_error,
            'symmetry_ops':np.stack(compatible_ops)}


def assign_wyckoff(data, catalog, seed, minimum_distance=0.25, retries=200):
    """Exact stoichiometry; finite special positions may only be occupied once."""
    rng = np.random.default_rng(seed)
    zs = data.atom_types.numpy()
    counts = Counter(int(z) for z in zs if z != 1)
    positions = catalog['positions']
    lattice = Lattice.from_parameters(*data.lengths.reshape(-1).tolist(),
                                      *data.angles.reshape(-1).tolist())

    def allocate(items, used, allocation, budget):
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        if not items:
            return allocation
        z, remaining = items[0]
        if remaining == 0:
            return allocate(items[1:], used, allocation, budget)
        choices = [i for i,p in enumerate(positions) if p['multiplicity'] <= remaining
                   and (p['dof'] > 0 or i not in used)]
        rng.shuffle(choices)
        for i in choices:
            pos = positions[i]
            result = allocate([(z,remaining-pos['multiplicity'])]+items[1:],
                              used | ({i} if pos['dof']==0 else set()),
                              allocation+[(z,i)],budget)
            if result is not None:
                return result
        return None

    for _ in range(retries):
        items = list(counts.items())
        rng.shuffle(items)
        allocation = allocate(items,set(),[],[50000])
        if allocation is None:
            raise ValueError(f'No compatible Wyckoff combination found for {counts}')
        coords = rng.random((len(zs),3))
        ops = np.tile(np.eye(4),(len(zs),1,1))
        anchors = np.arange(len(zs))
        available = {z:list(np.flatnonzero(zs==z)) for z in counts}
        metadata=[]
        all_heavy=[]
        valid=True
        for z,i in allocation:
            pos=positions[i]
            trial=rng.random(3)
            expanded=np.einsum('nij,j->ni',pos['ops'],np.r_[trial,1])[:,:3]
            distances=lattice.get_all_distances(expanded,expanded)
            np.fill_diagonal(distances,np.inf)
            if distances.min(initial=np.inf)<minimum_distance:
                valid=False;break
            if all_heavy and lattice.get_all_distances(expanded,np.array(all_heavy)).min()<minimum_distance:
                valid=False;break
            slots=available[z][:pos['multiplicity']]
            del available[z][:pos['multiplicity']]
            coords[slots]=expanded
            ops[slots]=pos['ops']
            anchors[slots]=slots[0]
            all_heavy.extend(expanded)
            metadata.append({'atomic_number':z,'wyckoff_letter':pos['letter'],
                             'standard_multiplicity':pos['standard_multiplicity'],
                             'input_cell_multiplicity':pos['multiplicity'],
                             'degrees_of_freedom':pos['dof'],'site_indices':slots})
        if not valid:
            continue
        # Orbit closure is checked under EVERY given input-cell SG operation.
        for z in counts:
            sites=coords[zs==z]
            for sym in catalog['symmetry_ops']:
                images=(sites @ sym[:3,:3].T+sym[:3,3]) % 1
                if lattice.get_all_distances(images,sites).min(axis=1).max()>1e-5:
                    raise ValueError('Allocated coordinates violate input space-group operations')
        # The first orbit map must leave the representative unchanged.
        regenerated=np.einsum('nij,nj->ni',ops,np.c_[coords[anchors],np.ones(len(zs))])[:,:3]
        delta=regenerated-coords;delta-=np.round(delta)
        if np.linalg.norm(delta[zs!=1] @ lattice.matrix,axis=1).max(initial=0)>1e-5:
            raise ValueError('Orbit representative maps are inconsistent')
        result=Data(frac_coords=torch.tensor(coords,dtype=torch.float),
                    atom_types=data.atom_types.clone(),lengths=data.lengths.clone(),angles=data.angles.clone(),
                    num_atoms=len(zs),num_nodes=len(zs),source_row=data.source_row.clone(),
                    spacegroup=torch.tensor([catalog['spacegroup']]),
                    ops=torch.tensor(ops,dtype=torch.float),anchor_index=torch.tensor(anchors),
                    ops_inv=torch.linalg.pinv(torch.tensor(ops[:,:3,:3],dtype=torch.float)))
        # Convert numpy site indices to ordinary ints for reproducible JSON records.
        for item in metadata:
            item['site_indices']=[int(x) for x in item['site_indices']]
        return result,metadata
    raise ValueError('Unable to place the selected orbit combinations without overlap')


def orbit_residual(coords, data):
    """Check maps across equivalent periodic branches of special positions."""
    xyz=np.asarray(coords)
    ops=data.ops.detach().cpu().numpy()
    anchors=data.anchor_index.detach().cpu().numpy()
    zs=data.atom_types.detach().cpu().numpy()
    shifts=np.array(list(product((-1,0,1),repeat=3)))
    maximum=0.0
    for anchor in np.unique(anchors[zs!=1]):
        indices=np.flatnonzero((anchors==anchor)&(zs!=1))
        first=ops[anchor]
        trial=xyz[anchor]+shifts
        representatives=trial @ first[:3,:3].T+first[:3,3]
        images=np.einsum('nij,sj->sni',ops[indices,:3,:3],representatives)+ops[indices,:3,3][None,:,:]
        delta=images-xyz[indices][None,:,:];delta-=np.round(delta)
        best=float(np.max(np.abs(delta),axis=(1,2)).min())
        maximum=max(maximum,best)
    return maximum
