"""Normalized operator-guided SMC with an explicit complex acquisition."""
import math
import torch

def real(x):return torch.cat([x.real.flatten(-2),x.imag.flatten(-2)],-1)
def complex_image(r,shape):
    n=shape[0]*shape[1];return torch.complex(r[...,:n],r[...,n:]).reshape(*r.shape[:-1],*shape)
class Acquisition:
    def __init__(self,mask,sens):self.mask=mask.bool();self.sens=sens;self.shape=mask.shape
    def forward(self,x):
        k=torch.fft.fft2(x[...,None,:,:]*self.sens,norm='ortho');return k[...,self.mask].flatten(-2)
    def adjoint(self,y):
        k=y.new_zeros(*y.shape[:-1],len(self.sens),*self.shape);k[...,self.mask]=y.reshape(*y.shape[:-1],len(self.sens),-1)
        return (torch.fft.ifft2(k,norm='ortho')*self.sens.conj()).sum(-3)
    def measure_real(self,x):
        y=self.forward(x);return torch.cat([y.real,y.imag],-1)

class FrozenModels:
    """Training-only PCA autoencoder, ridge linear operator, fitted linear noise predictors."""
    def __init__(self,images,physics,alpha,d=4,seed=31):
        self.shape=physics.shape;self.alpha=alpha;self.operator_calls=0
        x=real(images);self.mean=x.mean(0);_,_,vh=torch.linalg.svd(x-self.mean,full_matrices=False);self.basis=vh[:d]
        latent=(x-self.mean)@self.basis.T;self.scale=latent.std(0).clamp_min(.05);z=latent/self.scale
        zf=real(physics.adjoint(physics.forward(images)));self.op_mean=zf.mean(0);inp=zf-self.op_mean
        self.op_weight=torch.linalg.solve(inp.T@inp+.1*torch.eye(inp.shape[1]),inp.T@z)
        self.noise={};g=torch.Generator().manual_seed(seed)
        zz=z.repeat(64,1)
        for t in range(1,len(alpha)):
            eps=torch.randn(zz.shape,generator=g);v=alpha[t].sqrt()*zz+(1-alpha[t]).sqrt()*eps
            aug=torch.cat([v,torch.ones(len(v),1)],1)
            self.noise[t]=torch.linalg.solve(aug.T@aug+.1*torch.eye(d+1),aug.T@eps).detach()
        for value in self.__dict__.values():
            if isinstance(value,torch.Tensor):value.requires_grad_(False)
    def encode(self,x):return ((real(x)-self.mean)@self.basis.T)/self.scale
    def decode(self,z):return complex_image((z*self.scale)@self.basis+self.mean,self.shape)
    def operator_center(self,y,physics):
        self.operator_calls+=1
        # The operator outputs D(z); encoding that output gives the same standardized z.
        return self.encode(self.decode((real(physics.adjoint(y))-self.op_mean)@self.op_weight))
    def eps(self,z,t):return torch.cat([z,torch.ones(*z.shape[:-1],1)],-1)@self.noise[t]
    def z0(self,z,t):return (z-(1-self.alpha[t]).sqrt()*self.eps(z,t))/self.alpha[t].sqrt()

def log_normal(x,mean,var):
    if bool(torch.any(torch.as_tensor(var)<=0)):raise ValueError('Gaussian variance must be positive')
    var=torch.as_tensor(var,dtype=x.dtype)
    return -.5*(((x-mean).square()/var)+torch.log(var)+math.log(2*math.pi)).sum(-1)

def mixture_logpdf(x,base,bvar,center,ovar,rho):
    if not 0<=rho<1:raise ValueError('retained base component requires 0<=rho<1')
    b=log_normal(x,base,bvar)
    if rho==0:return b
    return torch.logaddexp(b+math.log1p(-rho),log_normal(x,center,ovar)+math.log(rho))

def normalize(logw):
    if not torch.isfinite(logw).all():raise FloatingPointError('nonfinite density; no silent uniform fallback')
    logw=logw-torch.logsumexp(logw,0);w=logw.exp();return w,float(1/w.square().sum())

class Sampler:
    def __init__(self,models,physics,y,config):
        self.m=models;self.A=physics;self.y=y;self.yr=torch.cat([y.real,y.imag]);self.c=config
        self.alpha=models.alpha;self.T=len(self.alpha)-1;self.d=models.basis.shape[0]
        self.center=models.operator_center(y,physics).detach() # exactly once per input
    def prior(self,z,t):
        if t==0:return self.m.z0(z,1),self.c['endpoint_variance']
        a=self.alpha;var=self.c['eta']**2*(1-a[t])/(1-a[t+1])*(1-a[t+1]/a[t])
        mean=a[t].sqrt()*self.m.z0(z,t+1)+(1-a[t]-var).clamp_min(0).sqrt()*self.m.eps(z,t+1)
        return mean,var
    def logL(self,z,t,u):
        obs=self.yr if t==0 else u[t];var=self.c['noise_real_variance'] if t==0 else self.c['aux_variance']*(1+t)
        return log_normal(obs,self.A.measure_real(self.m.decode(z)),var)
    def logh(self,z,t):
        return log_normal(self.yr,self.A.measure_real(self.m.decode(self.m.z0(z,t))),self.c['lookahead_variance']*(1+t))
    def proposal(self,ancestor,t,u):
        prior,var=self.prior(ancestor,t)
        v=ancestor.detach().requires_grad_(True)
        gy=torch.autograd.grad(-self.logh(v,t+1).sum(),v)[0]
        if t:
            b=prior.detach().requires_grad_(True);gu=torch.autograd.grad(-self.logL(b,t,u).sum(),b)[0]
        else:gu=torch.zeros_like(prior)
        # Fixed componentwise clipping is a declared demo gradient normalization.
        base=prior-self.c['guidance']*gy.clamp(-5,5)-self.c['guidance']*gu.clamp(-5,5)
        a=self.alpha;coef=((1-a[t])/(1-a[t+1])).sqrt()
        center=a[t].sqrt()*self.center+coef*(ancestor-a[t+1].sqrt()*self.center)
        return prior.detach(),var,base.detach(),float(var)*self.c['base_variance_factor'],center.detach()
    def initial_path(self,g):
        path=torch.zeros(self.T+1,self.d);a=self.alpha;z0=self.center
        path[self.T]=a[-1].sqrt()*z0+(1-a[-1]).sqrt()*torch.randn(self.d,generator=g)
        for t in range(self.T-1,0,-1):
            _,var=self.prior(path[t+1],t);eps=(path[t+1]-a[t+1].sqrt()*z0)/(1-a[t+1]).sqrt()
            path[t]=a[t].sqrt()*z0+(1-a[t]-var).clamp_min(0).sqrt()*eps+float(var)**.5*torch.randn(self.d,generator=g)
        path[0]=z0;return path
    def auxiliaries(self,path,g):
        return {t:self.A.measure_real(self.m.decode(path[t]))+(self.c['aux_variance']*(1+t))**.5*torch.randn(len(self.yr),generator=g) for t in range(1,self.T+1)}
    def sweep(self,u,g):
        N=self.c['particles'];paths=torch.zeros(N,self.T+1,self.d);paths[:,self.T]=torch.randn(N,self.d,generator=g);gene=torch.arange(N)
        z=paths[:,self.T];logw=self.logL(z,self.T,u)+self.logh(z,self.T);w,ess=normalize(logw)
        normalized_logw=logw-torch.logsumexp(logw,0)
        logs=[{'t':self.T,'ess_before_resampling':ess,'distinct_T_ancestor_fraction':1.,'resampled':False}];density=[]
        for t in range(self.T-1,-1,-1):
            resample=ess<self.c['ess_fraction']*N
            if resample:
                ids=torch.multinomial(w,N,replacement=True,generator=g);paths=paths[ids].clone();gene=gene[ids]
                carry=torch.zeros(N) # log W_ancestor - log r_ancestor = 0
            else:carry=normalized_logw
            ancestor=paths[:,t+1]
            prior,qvar,base,bvar,center=self.proposal(ancestor,t,u);ovar=self.c['operator_variance'];rho=self.c['mixture_rho']
            choice=torch.rand(N,generator=g)<rho;noise=torch.randn(N,self.d,generator=g)
            z=torch.where(choice[:,None],center+ovar**.5*noise,base+bvar**.5*noise);paths[:,t]=z
            logq=mixture_logpdf(z,base,bvar,center,ovar,rho);logp=log_normal(z,prior,qvar);lhnext=self.logh(ancestor,t+1);ll=self.logL(z,t,u);lh=self.logh(z,t) if t else torch.zeros(N)
            logw=carry+ll+lh+logp-lhnext-logq;w,ess=normalize(logw);normalized_logw=logw-torch.logsumexp(logw,0)
            density.append({'t':t,'logL':ll.tolist(),'logh':lh.tolist(),'logp':logp.tolist(),'logh_parent':lhnext.tolist(),'logq_marginal':logq.tolist(),'operator_component':choice.tolist(),'parent_indices':ids.tolist() if resample else list(range(N))})
            logs.append({'t':t,'ess_before_resampling':ess,'distinct_T_ancestor_fraction':float(gene.unique().numel()/N),'resampled':resample})
        picked=int(torch.multinomial(w,1,generator=g));return paths,w,paths[picked].clone(),logs,density
    def run(self,seed):
        g=torch.Generator().manual_seed(seed);path=self.initial_path(g);alllogs=[]
        for j in range(self.c['rounds']):
            u=self.auxiliaries(path,g);paths,w,path,logs,density=self.sweep(u,g);alllogs.append(logs)
        x=self.m.decode(paths[:,0]);mag=(w[:,None,None]*x.abs()).sum(0)
        rss=(w[:,None,None]*(x[:,None]*self.A.sens).abs().square().sum(1).sqrt()).sum(0)
        energy=self.y.abs().square().sum();residual=float((w*(self.A.forward(x)-self.y).abs().square().sum(-1)).sum()/energy) if energy>0 else None
        return {'magnitude':mag,'rss':rss,'complex_particles':x,'weights':w,'paths':paths,'last_aux':u,'logs':alllogs,'density_terms':density,'residual':residual,'selected_path':path}
