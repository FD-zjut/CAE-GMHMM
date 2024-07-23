import torch

b = torch.tensor([1.,2.,3.,4.],requires_grad=True)
a = torch.zeros([3,4])
a[0,:] += b
a[1,:] += a[0,:]
a[2,:] += a[1,:] * 2
loss = torch.mean(a)
loss.backward()
print(a.shape)
print(b.grad)
print(a)