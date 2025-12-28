cs231n 

Updated 2025: 

Pytorch broadcasting; the numpy/pytorch convention of (3,) means what? The naive interpretation is it means 3 rows. That isn't true and it requires the user to understand how to use squeeze and unsqueeze to differentiate between (3,) and (3,1). For batches the format is (B,T,C) so (3,1) can become (B,3,1). Each use case is different, writing pytorch code as an array vs. batch. 

Joint, Conditional and Marginal Probabilities. Is an image a Joint, Conditional or Marginal probablity distribution? 





