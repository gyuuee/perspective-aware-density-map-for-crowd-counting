# perspective-aware_densitymap

This project introduces a new density map generation method that can generate perspective-aware density maps for crowd datasets.

该项目介绍了一种新的密度图生成方法，可以为人群数据集生成透视感知的密度图。

Compared with the current geometric adaptive kernel estimation method, the perspective-aware density maps can train the crowd counting network to obtain better results.

与目前的几何自适应核估计法相比，可以训练人群计数网络得到更好的结果。

This project code only provides a method for generating a perspective perception density map on the ShanghaiTech PartA dataset.

本项目代码仅列出在ShanghaiTech PartA数据集上的透视感知密度图生成方法。

It is worth noting that the initial Gaussian kernel standard deviation in the code is intuitively set to 25. This parameter can be adjusted, and better results may be obtained.

值得注意的是，代码中初始的高斯核标准差在直观感觉上被设置为25，该参数可以调节，也许能得到更好的结果

# Using
* First, you need to set the 'root' path in the code.
* Then, the code needs to read the image size information and point annotation information of the ShanghaiTech PartA dataset.

代码需要读取ShanghaiTech PartA 数据集的图片大小信息以及点标注信息。

* The code will generate a perspective-aware density map in the ground_truth folder. If the true value density map of the picture already exists, the processing of the perspective-aware  density map will be skipped.

代码会在ground_truth文件夹下生成透视感知密度图，如果该图片的真值密度图已存在，将会跳过对该图片透视感知密度图的处理。
   
# References
If you find the our method useful, please cite our paper. Thank you!

```
@misc{gu2020recurrent,
      title={Recurrent Distillation based Crowd Counting}, 
      author={Yue Gu and Wenxi Liu},
      year={2020},
      eprint={2006.07755},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

