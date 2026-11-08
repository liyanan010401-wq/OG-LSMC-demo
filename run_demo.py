import argparse,json,time,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import torch,numpy as np
from src.core import Acquisition,FrozenModels,Sampler
from src.data import generate

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 c=json.loads((ROOT/'config.json').read_text());torch.set_num_threads(1);torch.manual_seed(c['seed']);start=time.time();path=ROOT/'data/sample.npz'
 if not path.exists():generate(path,c['data_seed'])
 d=np.load(path);x=torch.from_numpy(d['images']);A=Acquisition(torch.from_numpy(d['mask']),torch.from_numpy(d['sens']));alpha=torch.tensor(c['alpha'])
 models=FrozenModels(x[d['split']=='train'],A,alpha,c['latent_dim'],c['seed']);target=x[d['split']=='test'][0];y=A.forward(target)
 # Independent real/imag noise each variance R0, total complex variance 2*R0.
 g=torch.Generator().manual_seed(c['seed']+1);y=y+c['noise_real_variance']**.5*torch.complex(torch.randn(y.shape,generator=g),torch.randn(y.shape,generator=g))
 sampler=Sampler(models,A,y,c);out=sampler.run(c['seed']+2)
 m={'status':'synthetic smoke test; NOT MRI benchmark or calibrated posterior','seconds':time.time()-start,'operator_calls':models.operator_calls,'real_noise_variance':c['noise_real_variance'],'synthetic_magnitude_mse':float((out['magnitude']-target.abs()).square().mean()),'acquired_weighted_residual':out['residual'],'ess_and_genealogy':out['logs'],'weights_sum':float(out['weights'].sum()),'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'training_patients':list(range(18)),'test_patient':23,'posterior_scope':'last cloud conditional on final auxiliary draw, finite N/J approximation','substitutions':['linear PCA autoencoder','ridge operator, not LMO scan/convolution','fitted linear denoisers','synthetic supplied coil map; no ESPIRiT calibration']}
 np.savez_compressed(args.output/'reconstruction.npz',magnitude=out['magnitude'].numpy(),rss=out['rss'].numpy(),complex_particles=out['complex_particles'].numpy(),weights=out['weights'].numpy(),reference=target.numpy(),measurements=y.numpy())
 (args.output/'metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8');(args.output/'particle_trace.json').write_text(json.dumps(out['density_terms']),encoding='utf-8');print(json.dumps({'status':m['status'],'seconds':m['seconds']}))
if __name__=='__main__':main()
