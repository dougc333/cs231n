cs231n 

Updated 2025: 

1) setup notes for using nsys and ncu with colab
2) setup notes for distributed T4 gcloud setup with Kubernetes. Other forms of provisioning won't work. 
3) Numpy and Pytorch broadcasting for hw1.
4) The original numpy.pages from 10y ago doesn't cover broadcasting or np.dot() which are necessary for pytorch batch NN. 

Pytorch broadcasting; the numpy/pytorch convention of (3,) means what? The naive interpretation is it means 3 rows. That isn't true and it requires the user to understand how to use squeeze and unsqueeze to differentiate between (3,) and (3,1). For batches the format is (B,T,C) so (3,1) can become (B,3,1). Each use case is different, writing pytorch code as an array vs. batch. 


4) Probability review
Joint, Conditional and Marginal Probabilities.  





