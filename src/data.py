from pathlib import Path
import numpy as np

def generate(path,seed=4202):
 rng=np.random.default_rng(seed);N,H,W=24,8,8;y,x=np.mgrid[:H,:W];images=[]
 for i in range(N):
  a=np.exp(-((x-rng.uniform(2.5,4.5))**2+(y-rng.uniform(2.5,4.5))**2)/rng.uniform(3,7))
  a+=rng.uniform(.1,.35)*np.cos(x*.7)*np.sin(y*.6);phase=.15*x+.08*y+rng.uniform(-.2,.2)
  images.append((a*np.exp(1j*phase)).astype(np.complex64))
 mask=np.zeros((H,W),bool);mask[[0,1,3,5,7],:]=True;sens=np.ones((1,H,W),np.complex64)
 np.savez_compressed(path,images=np.array(images),mask=mask,sens=sens,split=np.array(['train']*18+['dev']*5+['test']),patient_id=np.arange(N),seed=seed)
if __name__=='__main__':generate(Path(__file__).resolve().parents[1]/'data/sample.npz')
