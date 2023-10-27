原始数据集   carvana-image-masking-challenge.zip

解压后有用的文件  train_hq train_masks 这两个文件夹，因为只有这两个文件夹有标注，所以我们只用这两个文件夹

数据说明

1、train_hq 一共是5088个图像，每个汽车有16张图像，总共318个汽车
2、train_masks 和 train_hq 一一对应

划分自己的训练集 ，按照1:9划分训练集和验证集

挑出前30个汽车IMG(480张)放入验证集中，构建数据集 dataset/val_img_dir
挑出对应的前30个汽车MASK(480张)放入验证集中，构建数据集 dataset/val_mask_dir

剩余的train_hq总计288个汽车（4608张），重新命名为 dataset/train_img_dir
剩余的train_masks总计288个汽车（4608张），重新命名为dataset/train_mask_dir
