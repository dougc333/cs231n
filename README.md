cs231n 

Updated 2025: 

1) colab_cuda_nemo.ipynb shows nvidia nemo working in colab which required updating the cuda drivers. Running nvidia-smi returns v12.5 and nvcc --version returns v12.4. To align both of these version you have to reinstall the cuda drivers and redo the symlinks with this code
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
3) Numpy and Pytorch broadcasting for hw1.
4) The original numpy.pages from 10y ago doesn't cover broadcasting or np.dot() which are necessary for pytorch batch NN. 

Pytorch broadcasting; the numpy/pytorch convention of (3,) means what? The naive interpretation is it means 3 rows. That isn't true and it requires the user to understand how to use squeeze and unsqueeze to differentiate between (3,) and (3,1). For batches the format is (B,T,C) so (3,1) can become (B,3,1). Each use case is different, writing pytorch code as an array vs. batch. 


4) Probability review
Joint, Conditional and Marginal Probabilities.  





