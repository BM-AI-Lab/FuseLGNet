# FuseLGNet.pytorch

## CoorLGNet Framework
![image](https://github.com/BM-AI-Lab/FuseLGNet/blob/master/FuseLGNet.png)

### Set up
```
- python==3.7
- cuda==11.7

# other pytorch/timm version can also work

pip install torch==1.7.0 torchvision==0.8.1;
pip install timm==0.4.12;
pip install torchprofile;

```

### Data preparation

Dataset storage format:

```
│FuseLGNet/dataset/parkinson/front
├──parkinson/
   ├──11（1）.jpg
   ├── 11（2）.jpg
   ├── ......
   ├── ......
├──normal/
   ├──11（1）.jpg
   ├── 11（2）.jpg
   ├── ......
   ├── ......
```

#### Introduction to the use of the code


```
1. Set `--data-path` to the `dataset` folder absolute path in the `train.py` script
2. Set `--weights`, `--batch_size`, `--epochs`, `--weight_decay`, `--lr` and and other parameters in the `train.py` script
3. After setting the `--data-path` and parameters, you can start training using the `train.py` script 
4. Import the same model in the `predict.py` script as in the training script and set `model_weight_path` to the trained model weight path (saved in the weights folder by default)
5. In the `predict.py` script, set `img_path` to the absolute path of the image you want to predict
6. Set the weight path `model_weight_path` and the predicted image path `img_path` and you can use the `predict.py` script to make predictions

```

