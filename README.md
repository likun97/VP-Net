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

 @article{likun2020vp,
  title={VP-Net: An Interpretable Deep Network for Variational Pansharpening},
  author={Tian Xin, Kun Li, Zhongyuan Wang, and Jiayi Ma},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={},
  pages={},
  year={2021},
  doi={10.1109/TGRS.2021.3089868}
  publisher={IEEE}
}


