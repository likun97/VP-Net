# VP-Net
Implementation of "VP-Net: An Interpretable Deep Network for Variational Pansharpening" （TGRS 2021 In print）

![image](https://user-images.githubusercontent.com/26796531/123496038-b4940e00-d658-11eb-8b0e-92bd15f58f27.png)



# To test with the pretrained model:
Run "CUDA_VISIBLE_DEVICES=0 python test.py" to test the network.

Please first prepare the testing data shown as ./data.

# To  train:
Run "CUDA_VISIBLE_DEVICES=0 python train.py" to train the network.

You train your own network according to the unique training dataset.


If this work is helpful, please cite it as:

@article{xu2020sdpnet,
  title={SDPNet: A Deep Network for Pan-Sharpening With Enhanced Information Representation},
  author={Xu, Han and Ma, Jiayi and Shao, Zhenfeng and Zhang, Hao and Jiang, Junjun and Guo, Xiaojie},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2020},
  publisher={IEEE}
}
