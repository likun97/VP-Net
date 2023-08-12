![image](https://user-images.githubusercontent.com/26796531/124340069-39e16a80-dbe5-11eb-9452-6c55c1efde93.png)

# VP-Net
Implementation of VP-Net: An Interpretable Deep Network for Variational Pansharpening

<div align="center">    
<img src=flowchart.png width="1000" height="800" alt="mflowchart">
<!-- <img src=flowchart.png width="600" height="150" alt="mflowchart"> -->
</div>

# To Train:
Run "CUDA_VISIBLE_DEVICES=0 python train.py" to train the network.

# To Test:
Run "CUDA_VISIBLE_DEVICES=0 python test.py" to test the network.


running environment :
python=3.7, tensorflow-gpu=1.15.0.

If this work is helpful to you, please cite it as：

      @article{tian2021vp,
        title={VP-Net: An Interpretable Deep Network for Variational Pansharpening},
        author={Xin Tian, Kun Li, Zhongyuan Wang, and Jiayi Ma},
        journal={IEEE Transactions on Geoscience and Remote Sensing},
        volume={},
        pages={1-16},
        year={2021},
        doi={10.1109/TGRS.2021.3089868},
        publisher={IEEE}
      } 
