import numpy as np
from engine.tensor import Tensor, softmax_cross_entropy
from nn.model import MLP
from nn.optim import Adam

def test_masked_gradient_is_zero_and_adam_state_cleared():
    model=MLP(dims=(2,3,2),seed=3)
    opt=Adam(model.parameters(),lr=0.01,masked=model.masked_parameters())
    layer=model.layers[0]
    # Establish nonzero optimizer moments first.
    x=Tensor(np.array([[1.0,0.5],[-0.3,0.8]]))
    loss=softmax_cross_entropy(model(x),np.array([0,1]))
    loss.backward(); opt.step()
    i,j=0,0
    layer.mask[i,j]=False
    layer.weight.data[i,j]=0
    opt.reset_state_at_masked()
    opt.zero_grad()
    loss=softmax_cross_entropy(model(Tensor(np.array([[0.4,0.7],[0.1,-0.2]]))),np.array([1,0]))
    loss.backward()
    assert layer.weight.grad[i,j] == 0.0
    before=layer.weight.data[i,j]
    opt.step()
    assert layer.weight.data[i,j] == 0.0
    assert opt.m[id(layer.weight)][i,j] == 0.0
    assert opt.v[id(layer.weight)][i,j] == 0.0
    assert before == 0.0

def test_revived_connection_starts_with_clean_state():
    model=MLP(dims=(2,3,2),seed=4)
    opt=Adam(model.parameters(),masked=model.masked_parameters())
    layer=model.layers[0]; i,j=0,1
    layer.mask[i,j]=False; layer.weight.data[i,j]=0
    opt.m[id(layer.weight)][i,j]=123
    opt.v[id(layer.weight)][i,j]=456
    opt.reset_state_at_masked()
    # Explicit revival policy: caller activates it and moments are already reset.
    layer.mask[i,j]=True
    assert opt.m[id(layer.weight)][i,j] == 0
    assert opt.v[id(layer.weight)][i,j] == 0


def test_sparse_inference_matches_dense_masked_forward():
    model=MLP(dims=(2,4,3),seed=9)
    model.layers[0].mask[0,0]=False
    model.layers[0].weight.data[0,0]=0
    model.layers[1].mask[1,2]=False
    model.layers[1].weight.data[1,2]=0
    x=np.array([[0.2,0.7],[-0.3,0.1],[1.0,-0.5]])
    dense=model(Tensor(x)).data
    sparse=model(Tensor(x),sparse=True).data
    assert np.allclose(dense,sparse,atol=1e-12,rtol=1e-12)
