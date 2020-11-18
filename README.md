# perspective-aware_densitymap

This project introduces a new density map generation method that can generate perspective-aware density maps for crowd datasets.
该项目介绍了一种新的密度图生成方法，可以为人群数据集生成透视感知的密度图。

Compared with the current geometric adaptive kernel estimation method, the perspective-aware density maps can train the crowd counting network to obtain better results.
与目前的几何自适应核估计法相比，可以训练人群计数网络得到更好的结果。

This project code only provides a method for generating a perspective perception density map on the ShanghaiTech PartA dataset.
本项目代码仅列出在ShanghaiTech PartA数据集上的透视感知密度图生成方法。

It is worth noting that the initial Gaussian kernel standard deviation in the code is intuitively set to 25. This parameter can be adjusted, and better results may be obtained.
值得注意的是，代码中初始的高斯核标准差在直观感觉上被设置为25，该参数可以调节，也许能得到更好的结果
   
# References
If you find the our method useful, please cite our paper. Thank you!

@inproceedings{10.1145/3417188.3417195,
author = {Gu, Yue},
title = {Perspective-Aware Distillation-Based Crowd Counting},
year = {2020},
isbn = {9781450375481},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3417188.3417195},
doi = {10.1145/3417188.3417195},
abstract = {Recently, with the progress of deep learning technologies, crowd counting has been rapidly developed. In this work, we first introduce a perspective-aware density map generation method that is able to produce ground-truth density maps from point annotations to train crowd counting model to accomplish superior performance than prior density map generation techniques. Besides, leveraging our density map generation method, we propose an iterative distillation algorithm to progressively enhance our model with identical network structures. In experiments, we demonstrate that, with our simple convolutional neural network architecture strengthened by our proposed training algorithm, our model is able to outperform or be comparable with the state-of-the-art methods.},
booktitle = {Proceedings of the 2020 4th International Conference on Deep Learning Technologies (ICDLT)},
pages = {123–128},
numpages = {6},
keywords = {Crowd counting, Density map generation, Distillation},
location = {Beijing, China},
series = {ICDLT 2020}
}
