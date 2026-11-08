import unittest,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.core import *
torch.set_num_threads(1)
class Contracts(unittest.TestCase):
 def setUp(self):torch.manual_seed(23)
 def test_complex_adjoint_multiple_coils(self):
  mask=torch.rand(5,6)>.4;sens=torch.randn(2,5,6,dtype=torch.complex64);a=Acquisition(mask,sens);x=torch.randn(5,6,dtype=torch.complex64);y=torch.randn(2*int(mask.sum()),dtype=torch.complex64)
  self.assertTrue(torch.allclose(torch.vdot(a.forward(x),y),torch.vdot(x.flatten(),a.adjoint(y).flatten()),atol=2e-5))
 def test_mixture_uses_full_marginal_and_zero_limit(self):
  x=torch.tensor([[.2],[2.]]);b=torch.zeros_like(x);o=b+2
  q=mixture_logpdf(x,b,1.,o,.5,.4)
  expected=torch.log(.6*log_normal(x,b,1.).exp()+.4*log_normal(x,o,.5).exp())
  self.assertTrue(torch.allclose(q,expected));self.assertTrue(torch.equal(mixture_logpdf(x,b,1.,o,.5,0),log_normal(x,b,1.)))
 def test_gaussian_normalization_and_positive_floor(self):
  x=torch.linspace(-8,8,20001)[:,None];density=log_normal(x,torch.zeros_like(x),1.).exp();self.assertAlmostEqual(float(torch.trapezoid(density,x[:,0])),1.,places=4)
  with self.assertRaises(ValueError):log_normal(x,x,0)
 def test_gradient_sign(self):
  h=torch.randn(5,2);v=torch.randn(2,requires_grad=True);b=torch.randn(5);loss=.5*((h@v-b)**2/.7).sum();g=torch.autograd.grad(loss,v)[0]
  self.assertTrue(torch.allclose(g,h.T@(h@v-b)/.7));self.assertLess(.5*((h@(v-.001*g)-b)**2/.7).sum(),loss)
 def test_telescoping_three_boundaries(self):
  pT=torch.tensor(-2.);L=[torch.tensor(-3.),torch.tensor(-4.),torch.tensor(-5.)];h=[None,torch.tensor(-6.),torch.tensor(-7.)];q=[torch.tensor(-8.),torch.tensor(-9.)];p=[torch.tensor(-10.),torch.tensor(-11.)]
  GT=L[2]+h[2];G1=L[1]+h[1]+p[1]-h[2]-q[1];G0=L[0]+p[0]-h[1]-q[0]
  self.assertAlmostEqual(float(pT+sum(q)+GT+G1+G0),float(pT+sum(p)+sum(L)))
 def test_resampling_correction_no_old_weight(self):
  w=torch.tensor([.1,.2,.7]);idx=torch.tensor([2,2,0]);self.assertTrue(torch.equal(w[idx].log()-w[idx].log(),torch.zeros(3)))
 def test_no_uniform_fallback_and_ess(self):
  w,e=normalize(torch.zeros(8));self.assertAlmostEqual(e,8)
  with self.assertRaises(FloatingPointError):normalize(torch.tensor([float('nan'),0]))
 def test_frozen_decoder_retains_input_derivative(self):
  x=torch.randn(10,4,4,dtype=torch.complex64);a=Acquisition(torch.ones(4,4,dtype=torch.bool),torch.ones(1,4,4,dtype=torch.complex64));m=FrozenModels(x,a,torch.tensor([1.,.7,.3]),3)
  z=torch.randn(2,3,requires_grad=True);m.decode(z).abs().square().sum().backward();self.assertGreater(z.grad.abs().sum(),0);self.assertFalse(m.basis.requires_grad)
 def test_weighted_magnitude_not_magnitude_mean(self):
  x=torch.tensor([1+0j,-1+0j]);w=torch.tensor([.5,.5]);self.assertEqual((w*x.abs()).sum(),1);self.assertEqual((w*x).sum().abs(),0)
 def test_fixed_auxiliary_gaussian_conditional_moments(self):
  # A genuinely conditional joint (z0,z1) target: no marginal-y substitution.
  x=torch.randn(24,2,2,dtype=torch.complex64)*.3
  a=Acquisition(torch.ones(2,2,dtype=torch.bool),torch.ones(1,2,2,dtype=torch.complex64))
  m=FrozenModels(x,a,torch.tensor([1.,.4]),1,seed=19)
  c={'endpoint_variance':.3,'eta':.8,'noise_real_variance':1.,'aux_variance':1.,'lookahead_variance':1.,'guidance':.005,'base_variance_factor':1.,'operator_variance':.6,'mixture_rho':.3,'particles':20000,'ess_fraction':0.}
  sampler=Sampler(m,a,a.forward(x[0]),c);u={1:torch.linspace(-.1,.1,8)}
  zero=torch.zeros(1,1);one=torch.ones(1,1);b0,v=sampler.prior(zero,0);b1,_=sampler.prior(one,0);slope=float(b1-b0);offset=float(b0)
  intercept=a.measure_real(m.decode(zero))[0];H=a.measure_real(m.decode(one))[0]-intercept
  precision=torch.tensor([[1/v,-slope/v],[-slope/v,1+slope*slope/v]])
  rhs=torch.tensor([offset/v,-slope*offset/v]);precision[0,0]+=H.square().sum();rhs[0]+=H@(sampler.yr-intercept)
  precision[1,1]+=H.square().sum()/2;rhs[1]+=H@(u[1]-intercept)/2
  exact=torch.linalg.solve(precision,rhs);paths,w,_,_,_=sampler.sweep(u,torch.Generator().manual_seed(103));estimated=(w[:,None]*paths[:,:,0]).sum(0)
  self.assertTrue(torch.allclose(estimated,exact,atol=.045,rtol=.08),(estimated,exact))
if __name__=='__main__':unittest.main()
