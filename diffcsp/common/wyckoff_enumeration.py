"""Enumerate every labelled Wyckoff multiset, without random scheme selection."""
from collections import Counter
import numpy as np
import torch
from pymatgen.core import Lattice
from torch_geometric.data import Data


def enumerate_allocations(counts,positions):
    elements=sorted(counts)
    def element(k,used,allocation):
        if k==len(elements):
            yield tuple(allocation)
            return
        z=elements[k]
        def partition(remaining,first,occupied,items):
            if remaining==0:
                yield from element(k+1,occupied,allocation+items)
                return
            for i in range(first,len(positions)):
                p=positions[i]
                if p['multiplicity']>remaining or (p['dof']==0 and i in occupied):continue
                yield from partition(remaining-p['multiplicity'],i if p['dof']>0 else i+1,
                                     occupied|({i} if p['dof']==0 else set()),items+[(z,i)])
        yield from partition(counts[z],0,used,[])
    yield from element(0,set(),[])


def instantiate_allocation(data,catalog,allocation,seed,minimum_distance=.25,retries=200):
    """Initialize coordinates for THIS exact allocation; never replace its orbits."""
    zs=data.atom_types.numpy();positions=catalog['positions']
    lattice=Lattice.from_parameters(*data.lengths.reshape(-1).tolist(),*data.angles.reshape(-1).tolist())
    rng=np.random.default_rng(seed)
    for _ in range(retries):
        xyz=rng.random((len(zs),3));ops=np.tile(np.eye(4),(len(zs),1,1));anchors=np.arange(len(zs))
        available={int(z):list(np.flatnonzero(zs==z)) for z in set(zs) if z!=1}
        metadata=[]
        for z,i in allocation:
            p=positions[i];slots=available[z][:p['multiplicity']]
            del available[z][:p['multiplicity']]
            if len(slots)!=p['multiplicity']:raise ValueError('Allocation exceeds element count')
            xyz[slots]=np.einsum('nij,j->ni',p['ops'],np.r_[rng.random(3),1])[:,:3]
            ops[slots]=p['ops'];anchors[slots]=slots[0]
            metadata.append({'atomic_number':int(z),'wyckoff_letter':p['letter'],
                             'standard_multiplicity':int(p['standard_multiplicity']),
                             'input_cell_multiplicity':int(p['multiplicity']),
                             'degrees_of_freedom':int(p['dof']),
                             'site_indices':[int(x) for x in slots]})
        if any(available.values()):raise ValueError('Allocation does not fill composition')
        heavy=np.flatnonzero(zs!=1)
        if minimum_distance>0:
            distances=lattice.get_all_distances(xyz[heavy],xyz[heavy]);np.fill_diagonal(distances,np.inf)
            if distances.min(initial=np.inf)<minimum_distance:continue
        for z in available:
            sites=xyz[zs==z]
            for sym in catalog['symmetry_ops']:
                images=sites@sym[:3,:3].T+sym[:3,3]
                if lattice.get_all_distances(images,sites).min(axis=1).max()>1e-5:
                    raise ValueError('Enumerated orbit fails space-group closure')
        return Data(frac_coords=torch.tensor(xyz,dtype=torch.float),atom_types=data.atom_types.clone(),
                    lengths=data.lengths.clone(),angles=data.angles.clone(),num_atoms=len(zs),num_nodes=len(zs),
                    source_row=data.source_row.clone(),spacegroup=torch.tensor([catalog['spacegroup']]),
                    ops=torch.tensor(ops,dtype=torch.float),anchor_index=torch.tensor(anchors),
                    ops_inv=torch.linalg.pinv(torch.tensor(ops[:,:3,:3],dtype=torch.float))),metadata
    raise ValueError('Could not initialize this specific allocation without overlap; not substituted')
