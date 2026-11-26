"""Fast fixed-cell powder-profile fitting in Wyckoff free coordinates."""
from dataclasses import dataclass
import numpy as np
from ase.data import covalent_radii
from pymatgen.core import Element
from pymatgen.analysis.diffraction.xrd import ATOMIC_SCATTERING_PARAMS
from scipy.optimize import minimize
from scipy.sparse import csr_matrix


class FixedCellPXRD:
    def __init__(self,lattice,wavelength=1.5406,two_theta=(5.,80.),step=.04,fwhm=.15):
        self.lattice=lattice
        lower,upper=two_theta
        reciprocal=lattice.reciprocal_lattice_crystallographic
        maximum=2*np.sin(np.deg2rad(upper/2))/wavelength
        points=reciprocal.get_points_in_sphere([[0,0,0]],[0,0,0],maximum)
        rows=[]
        for point,length,*_ in points:
            if length<1e-10:continue
            theta=np.arcsin(np.clip(wavelength*length/2,0,1))
            angle=float(np.rad2deg(2*theta))
            if angle>=lower-1e-8:
                rows.append((angle,np.rint(point).astype(int),length,theta))
        rows.sort(key=lambda x:x[0])
        if not rows:raise ValueError('No reflections in angle range')
        self.hkls=np.array([r[1] for r in rows])
        self.s2=np.array([(r[2]/2)**2 for r in rows])
        self.lp=np.array([(1+np.cos(2*r[3])**2)/(np.sin(r[3])**2*np.cos(r[3])) for r in rows])
        centers=[];groups=[]
        for angle,*_ in rows:
            if not centers or angle-centers[-1]>1e-5:centers.append(angle)
            groups.append(len(centers)-1)
        self.groups=np.array(groups);self.centers=np.array(centers)
        self.grid=np.arange(lower,upper+step/2,step)
        sigma=fwhm/(2*np.sqrt(2*np.log(2)))
        rr=[];cc=[];vv=[]
        for j,center in enumerate(centers):
            selected=np.flatnonzero(np.abs(self.grid-center)<=4*sigma)
            rr.extend(selected);cc.extend([j]*len(selected))
            vv.extend(np.exp(-.5*((self.grid[selected]-center)/sigma)**2))
        self.broadening=csr_matrix((vv,(rr,cc)),shape=(len(self.grid),len(centers)))
        self.gram=(self.broadening.T@self.broadening).tocsr()
        self._factors={}

    def scattering(self,zs):
        columns=[]
        for z in zs:
            z=int(z)
            if z not in self._factors:
                coefficients=np.array(ATOMIC_SCATTERING_PARAMS[Element.from_Z(z).symbol])
                self._factors[z]=z-41.78214*self.s2*np.sum(coefficients[None,:,0]*np.exp(-self.s2[:,None]*coefficients[None,:,1]),axis=1)
            columns.append(self._factors[z])
        return np.stack(columns,axis=1)

    def peaks(self,zs,xyz):
        factors=self.scattering(zs)
        amplitude=np.sum(factors*np.exp(2j*np.pi*(self.hkls@np.asarray(xyz).T)),axis=1)
        return np.bincount(self.groups,weights=np.abs(amplitude)**2*self.lp,minlength=len(self.centers))

    def profile(self,peaks):
        y=np.asarray(self.broadening@peaks).ravel()
        return y/max(float(y.max()),1e-30)

    def objective(self,xyz,derivatives,zs,target):
        """1-cosine on Gaussian-broadened powder profiles, exact analytic gradient."""
        wave=self.scattering(zs)*np.exp(2j*np.pi*(self.hkls@xyz.T))
        amplitude=wave.sum(axis=1)
        intensities=np.abs(amplitude)**2*self.lp
        peaks=np.bincount(self.groups,weights=intensities,minlength=len(self.centers))
        mp=np.asarray(self.gram@peaks).ravel();mt=np.asarray(self.gram@target).ravel()
        pp=max(float(peaks@mp),1e-30);tt=max(float(target@mt),1e-30)
        denominator=np.sqrt(pp*tt)
        cosine=float(peaks@mt)/denominator
        grad_peaks=cosine*mp/pp-mt/denominator
        reflection_gradient=grad_peaks[self.groups]*self.lp
        # dF/dq = 2 pi i sum_j f_j exp(i phase_j) h.C_j .
        phase_derivative=np.einsum('ri,aip->rap',self.hkls,derivatives,optimize=True)
        d_amplitude=2j*np.pi*np.einsum('ra,rap->rp',wave,phase_derivative,optimize=True)
        d_intensity=2*np.real(amplitude.conjugate()[:,None]*d_amplitude)
        gradient=np.einsum('r,rp->p',reflection_gradient,d_intensity,optimize=True)
        return 1-cosine,gradient


@dataclass
class OrbitCoordinates:
    base:np.ndarray
    derivatives:np.ndarray
    initial:np.ndarray
    species:np.ndarray

    @classmethod
    def from_assignment(cls,data,metadata):
        zs=data.atom_types.numpy();heavy=np.flatnonzero(zs!=1)
        lookup={int(site):i for i,site in enumerate(heavy)}
        ops=data.ops.numpy().astype(float);coords=data.frac_coords.numpy()
        dofs=[]
        for item in metadata:
            anchor=item['site_indices'][0]
            u,s,_=np.linalg.svd(ops[anchor,:3,:3])
            basis=u[:,s>1e-6]
            dofs.append((item,anchor,basis))
        total=sum(x[2].shape[1] for x in dofs)
        base=np.zeros((len(heavy),3));c=np.zeros((len(heavy),3,total));initial=[];offset=0
        for item,anchor,basis in dofs:
            width=basis.shape[1];bias=ops[anchor,:3,3]
            initial.extend(basis.T@(coords[anchor]-bias))
            for site in item['site_indices']:
                j=lookup[site]
                base[j]=ops[site,:3,:3]@bias+ops[site,:3,3]
                c[j,:,offset:offset+width]=ops[site,:3,:3]@basis
            offset+=width
        result=cls(base,c,np.array(initial),zs[heavy])
        delta=result.coordinates(result.initial)-coords[heavy]
        delta-=np.round(delta)
        if np.max(np.abs(delta))>2e-5:raise ValueError('Free-coordinate parameterization mismatch')
        return result

    def coordinates(self,q):return self.base+np.einsum('aip,p->ai',self.derivatives,q)


def distances_valid(lattice,zs,xyz,scale=.6):
    radius=covalent_radii[np.asarray(zs,dtype=int)]
    distance=lattice.get_all_distances(xyz,xyz)
    np.fill_diagonal(distance,min(lattice.get_niggli_reduced_lattice().abc))
    ratio=float(np.min(distance/(radius[:,None]+radius[None,:])))
    return ratio+1e-8>=scale,ratio


def unavoidable_overlap(data,metadata,lattice,scale=.6):
    """The same static rules as the preceding combinatorial overlap screening."""
    zs=data.atom_types.numpy();ops=data.ops.numpy()
    shortest=min(lattice.get_niggli_reduced_lattice().abc)
    fixed=[]
    for item in metadata:
        indices=item['site_indices'];z=item['atomic_number'];cut=2*scale*covalent_radii[z]
        if shortest+1e-8<cut:return True
        for ai,i in enumerate(indices):
            for j in indices[:ai]:
                if np.allclose(ops[i,:3,:3],ops[j,:3,:3],atol=1e-7):
                    if lattice.get_distance_and_image(ops[i,:3,3],ops[j,:3,3])[0]+1e-8<cut:return True
        if item['degrees_of_freedom']==0:fixed.extend(indices)
    if len(fixed)>1:
        distance=lattice.get_all_distances(ops[fixed,:3,3],ops[fixed,:3,3])
        radius=covalent_radii[zs[fixed]];np.fill_diagonal(distance,np.inf)
        if np.any(distance+1e-8<scale*(radius[:,None]+radius[None,:])):return True
    return False


def fit_scheme(engine,data,metadata,target,starts=3,maxiter=60,seed=42,scale=.6):
    orbit=OrbitCoordinates.from_assignment(data,metadata)
    rng=np.random.default_rng(seed);results=[]
    for start in range(starts if len(orbit.initial) else 1):
        q=orbit.initial.copy()
        if start:q+=rng.uniform(-1,1,len(q))
        initial_loss,_=engine.objective(orbit.coordinates(q),orbit.derivatives,orbit.species,target)
        if len(q):
            optimum=minimize(lambda parameters:engine.objective(orbit.coordinates(parameters),orbit.derivatives,orbit.species,target),
                             q,jac=True,method='L-BFGS-B',bounds=[(x-1,x+1) for x in q],
                             options={'maxiter':maxiter,'maxfun':maxiter*3,'ftol':1e-9,'gtol':1e-6})
            best=optimum.x;success=bool(optimum.success);nfev=int(optimum.nfev)
        else:best=q;success=True;nfev=1
        xyz=orbit.coordinates(best)
        loss,_=engine.objective(xyz,orbit.derivatives,orbit.species,target)
        valid,ratio=distances_valid(engine.lattice,orbit.species,xyz,scale)
        results.append({'start':start+1,'initial_cosine':float(1-initial_loss),
                        'initial_parameters':q.tolist(),
                        'cosine':float(1-loss),'optimizer_converged':success,'function_evaluations':nfev,
                        'geometry_passed':valid,'min_distance_ratio':ratio,'parameters':best.tolist()})
    valid_results=[x for x in results if x['geometry_passed']]
    best=max(valid_results or results,key=lambda x:x['cosine'])
    return {'starts':results,'best':best,'free_parameters':len(orbit.initial),
            'coordinates':orbit.coordinates(np.array(best['parameters'])),
            'initial_coordinates':orbit.coordinates(np.array(best['initial_parameters'])),
            'species':orbit.species}
