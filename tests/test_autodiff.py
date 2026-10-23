import numpy as np
from engine.tensor import Tensor, softmax_cross_entropy

def finite_difference(fn, x, eps=1e-6):
    grad=np.zeros_like(x,dtype=float)
    for idx in np.ndindex(x.shape):
        xp=x.copy(); xm=x.copy(); xp[idx]+=eps; xm[idx]-=eps
        grad[idx]=(fn(xp)-fn(xm))/(2*eps)
    return grad

def check(fn, x, atol=2e-5):
    t=Tensor(x,requires_grad=True); out=fn(t); out.backward()
    numeric=finite_difference(lambda z: float(fn(Tensor(z)).data), np.array(x,dtype=float))
    assert np.allclose(t.grad,numeric,atol=atol,rtol=2e-4), (t.grad,numeric)

def test_add_broadcast():
    x=np.arange(6,dtype=float).reshape(2,3)/7
    b=np.array([.2,-.3,.5])
    tx=Tensor(x,True); tb=Tensor(b,True); loss=((tx+tb)*2).mean(); loss.backward()
    assert np.allclose(tb.grad,np.full(3,2/3))
    assert np.allclose(tx.grad,np.full((2,3),1/3))

def test_sub_mul_div_sum_mean_relu_tanh_sigmoid():
    x=np.array([.3,.7,1.2])
    check(lambda t: ((t-0.4)*(t+1.2)/ (t+2.0)).sum(),x)
    check(lambda t: (t*t).mean(),x)
    check(lambda t: t.relu().sum(),x)
    check(lambda t: t.tanh().sum(),x)
    check(lambda t: t.sigmoid().sum(),x)

def test_matmul_gradient():
    rng=np.random.default_rng(1); a=rng.normal(size=(2,3)); b=rng.normal(size=(3,2))
    ta=Tensor(a,True); tb=Tensor(b,True); loss=((ta@tb)*(ta@tb)).mean(); loss.backward()
    na=finite_difference(lambda z: float(((Tensor(z)@Tensor(b)).data**2).mean()),a)
    nb=finite_difference(lambda z: float(((Tensor(a)@Tensor(z)).data**2).mean()),b)
    assert np.allclose(ta.grad,na,atol=2e-5,rtol=2e-4)
    assert np.allclose(tb.grad,nb,atol=2e-5,rtol=2e-4)

def test_softmax_cross_entropy_gradient():
    z=np.array([[.2,-.4,.7],[1.2,.3,-.1]])
    y=np.array([2,0])
    tz=Tensor(z,True); loss=softmax_cross_entropy(tz,y); loss.backward()
    nz=finite_difference(lambda a: float(softmax_cross_entropy(Tensor(a),y).data),z)
    assert np.allclose(tz.grad,nz,atol=2e-5,rtol=2e-4)

def test_gradient_accumulation_shared_node():
    x=Tensor(np.array([.7,1.1]),True)
    y=x*x + x
    y.sum().backward()
    assert np.allclose(x.grad,2*x.data+1)
