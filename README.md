cs231n 

Updated 2025: 

1) colab_cuda_nemo.ipynb shows nvidia nemo working in colab which required updating the cuda drivers. Running nvidia-smi returns v12.5 and nvcc --version returns v12.4. To align both of these version you have to reinstall the cuda drivers and redo the symlinks with this code. Nemo presents other problems besides cuda driver versions. Have to install teh github Mamba state space repo; and there are plethora of error messages for the unmaintained Mamba github; but it works. 
```
apt-get update && apt-get upgrade
apt-get install emacs
export  DEBIAN_FRONTEND=noninteractive 
apt-get install keyboard-configuration
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get -y install cuda-toolkit-12-4
rm /etc/alternatives/cuda
ln -s  /usr/local/cuda-12.4 /etc/alternatives/cuda
nvcc --version 
```

2) setup notes for distributed T4 gcloud setup with Kubernetes. Other forms of provisioning won't work, ie requesting a T4 instance from ease1-f will give out of resources messages. They confgured the cluster so you can only get T4 through GKE. 
3) Numpy and Pytorch broadcasting. Pytorch broadcasting; the numpy/pytorch convention of (3,) means what? The naive interpretation is it means 3 rows. That isn't true and it requires the user to understand how to use squeeze and unsqueeze to differentiate between (3,) and (3,1). For batches the format is (B,T,C) so (3,1) can become (B,3,1). Each use case is different, writing pytorch code as an array vs. batch.
```
# test examples with pytorch

#Broadcast vector to scalar
print('vector scalar broadcasting')
a = np.array([1,1,1])
b = np.array([3])
print(a.shape, b.shape)
# right->left align (3,), (1,) means 1 gets expanded to 3


print('--------------')
print('matrix broadcasting')
print('careful matrix broadcasting rules different than scalar')
print('--------------')

A = np.ones((10, 3, 4))
B = np.ones((1, 4, 5))

C = A @ B
print(C.shape)

(3,) (1,)
--------------
(10, 3, 5)
```

5) np.dot() doesn't broadcast! do not use. np.dot() supports both dot product and matrix mul. If it doesn't support broadcast then the matmul is broken for some data. Data dependent error. np.dot(a,b) != a@b
```
a = np.random.rand(3,4,2)
b = np.random.rand(2,4)
d = np.dot(a,b)
e = a@b
print(d==e)

[[[ True  True  True  True]
  [ True  True  True False]
  [ True  True  True  True]
  [ True False  True  True]]

 [[ True  True  True  True]
  [False  True  True False]
  [ True  True  True  True]
  [False False  True  True]]

 [[ True  True  True  True]
  [False  True  True False]
  [ True  True  True  True]
  [ True  True  True  True]]]
```
The result should be all True.

7) the issue with softmax() is the performance hit a transpose causes. Added transpose document.  
8) Probability review. Joint, Conditional and Marginal Probabilities and implications for NN> 





