# GAN实现

此代码用于实现基本GAN算法，并使用动漫头像数据集来实现一个动漫头像生成的小demo。

## 环境配置

提供requirements.txt文件用于环境配置，里面有所有需要的环境，可使用`conda env create -f requirements.txt`命令直接配置代码运行需要的环境。

## 数据说明

数据来自知乎用户[何之源](https://links.jianshu.com/go?to=https%3A%2F%2Fwww.zhihu.com%2Fpeople%2Fhe-zhi-yuan-16)爬取并经过处理的图片，共是51223张图片，尺寸是96×96×3，总大小272 MB。

所有数据已上传至睿客云盘，请通过[https://rec.ustc.edu.cn/share/a0efd5b0-f90f-11ec-a6a3-231c05f2c0ca](https://rec.ustc.edu.cn/share/a0efd5b0-f90f-11ec-a6a3-231c05f2c0ca)链接下载数据。

下载后文件存放格式如下：

```
-/
	-GAN.py
	-data/
		-faces/
	-README.md
	-report.md
	-report.pdf
	...
```

即将文件改名为data并放到GAN.py相同文件夹。