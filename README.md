# LBCNN
An implementation of LBCNN.

Paper: [Local Binary Convolutional Neural Networks](https://arxiv.org/abs/1608.06049)

Code Referance: [dizcza/lbcnn.pytorch](https://github.com/dizcza/lbcnn.pytorch)

I have tried [dizcza](https://github.com/dizcza/lbcnn.pytorch)'s code, but it didn't work. So I rewrite the LBC module.Based on the LBC module, I built a simple model and compared it with the classical CNNs model. 

I run my code on my laptop with CPU(core i5), only 1 epoch(>_<),here are the results.

## Model based on LBC
Layer1: (in_channel=1, out_channel=6, num_of_anchor_weight=4, sparsity=0.9, kernel_size=3, padding=1) -> MaxPool_2x2
Layer2: (in_channel=6, out_channel=16, num_of_anchor_weight=4, sparsity=0.9, kernel_size=3, padding=1) -> MaxPool_2x2
Full connection layer: fc(100) -> relu -> fc(10)
```
epoch 1, iter 100: loss 0.528, time: 10.600
epoch 1, iter 200: loss 0.473, time: 10.218
epoch 1, iter 300: loss 0.437, time: 10.514
epoch 1, iter 400: loss 0.222, time: 9.998
epoch 1, iter 500: loss 0.256, time: 10.560
epoch 1, iter 600: loss 0.262, time: 10.328
Test Accuracy of the model on the 10000 test images: 92.15 %
```


## Model based on CNN
Layer1: (in_channel=1, out_channel=6, kernel_size=3, padding=1) -> MaxPool_2x2
Layer2: (in_channel=6, out_channel=16, kernel_size=3, padding=1) -> MaxPool_2x2
Full connection layer: fc(100) -> relu -> fc(10)

```
epoch 1, iter 100: loss 0.870, time: 4.551
epoch 1, iter 200: loss 0.546, time: 4.497
epoch 1, iter 300: loss 0.438, time: 4.548
epoch 1, iter 400: loss 0.312, time: 4.474
epoch 1, iter 500: loss 0.389, time: 4.452
epoch 1, iter 600: loss 0.241, time: 4.498
Test Accuracy of the model on the 10000 test images: 90.53 %
```
