import json,sys,warnings
from pathlib import Path
import numpy as np
from pymatgen.io.cif import CifParser
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diffcsp.common.pxrd_wyckoff import FixedCellPXRD
warnings.filterwarnings('ignore')
def calc(path,cache,reference=False):
    s=CifParser(path,frac_tolerance=0,site_tolerance=1e-5 if reference else 1e-10).parse_structures(primitive=False)[0]
    key=tuple(s.lattice.matrix.ravel())
    if key not in cache:cache[key]=FixedCellPXRD(s.lattice)
    eng=cache[key];zs=[];xyz=[];occ=[]
    for site in s:
        for element,occupancy in site.species.items():
            zs.append(element.Z);xyz.append(site.frac_coords);occ.append(occupancy)
    factors=eng.scattering(zs)*np.array(occ)[None,:]
    amp=np.sum(factors*np.exp(2j*np.pi*(eng.hkls@np.array(xyz).T)),axis=1)
    peaks=np.bincount(eng.groups,weights=np.abs(amp)**2*eng.lp,minlength=len(eng.centers))
    return eng.profile(peaks),float(sum(o for z,o in zip(zs,occ) if z==1))
def cosine(a,b,mask):return float(a[mask]@b[mask]/(np.linalg.norm(a[mask])*np.linalg.norm(b[mask])))

manifest=json.loads((ROOT/'figures/ICDD_ref_wp_original_vs_generated_XRD/manifest.json').read_text(encoding='utf-8'))
old=ROOT/'results/icdd_wyckoff_top20_4gpu_hpc'
profiles=ROOT/'results/icdd_ref_wp_results_logs_20261027/results/original_vs_generated_xrd/profiles'
out=ROOT/'results/icdd_gallery_combined_xrd_20261027';out.mkdir(exist_ok=True)
rows=[];errors=[];all_candidates=[]
for i,r in enumerate(manifest):
 card=r['card'];z=np.load(profiles/(card+'.npz'));target=z['experimental'];mask=z['score_mask'].astype(bool);cache={}
 assert abs(cosine(z['reference'],target,mask)-r['reference_cosine'])<1e-9
 best=dict(score=r['best_cosine'],scheme=r['best_scheme'],run='reference_wyckoff_gallery',converged=r['converged'])
 for p in sorted((old/'relaxed_cifs'/card).glob('*.cif')):
  try:
   y,_=calc(p,cache);score=cosine(y,target,mask)
   status=json.loads((old/'relax_records'/card/(p.stem+'.json')).read_text())
   c=dict(card=card,score=score,scheme=p.stem,run='top20',converged=status.get('converged',False));all_candidates.append(c)
   if score>best['score']:best=c
  except Exception as e:errors.append(dict(card=card,scheme=p.stem,error=str(e)))
 rows.append(dict(card=card,original=r['reference_cosine'],generated_optimized=best['score'],new_run_cosine=r['best_cosine'],scheme=best['scheme'],run=best['run'],converged=best['converged']))
 if (i+1)%50==0:print(i+1,flush=True)
summary=dict(paired=len(rows),best_from_new=sum(r['run']=='reference_wyckoff_gallery' for r in rows),unconverged_selected=sum(not r['converged'] for r in rows),improved=sum(r['generated_optimized']>r['original'] for r in rows),original_median=float(np.median([r['original'] for r in rows])),generated_median=float(np.median([r['generated_optimized'] for r in rows])),selection='gallery protocol: highest cosine over all available optimized structures, no convergence/geometry/energy filter',source=str(ROOT/'figures/ICDD_ref_wp_original_vs_generated_XRD/manifest.json'))
for name,data in [('paired_values',rows),('summary',summary),('top20_scores',all_candidates),('errors',errors)]:
 (out/(name+'.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
