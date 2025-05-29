
## 环境准备
```bash
pip install torch torchvision tensorboard pillow numpy
```

## 数据集准备
1. 下载 Caltech-101 数据集
2. 在根目录下创建 `\caltech-101\101_ObjectCategories` 文件夹，并将数据放在这个文件夹下
3. 运行数据预处理脚本: `python prepare_data.py`
4. 在根目录下创建`data`文件夹，将生成的train、test放在文件夹放在中

## 训练流程
1. 在main.py中修改参数，运行:
```bash
python main.py --pretrained True --model resnet18
```

## 也可直接通过https://pan.baidu.com/s/1JGZl4N1C_lGC5NesMuhDMQ?pwd=d1kt 下载训练日志和结果

## 查看训练结果
```bash
python -m tensorboard.main --logdir=runs
```

## 文件说明
1.模型文件保存在`checkpoint`文件夹中
2.tensorbaoard日志保存在`runs`文件夹中
3.文件名为预设参数组合
