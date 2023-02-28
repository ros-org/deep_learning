
# 参考csdn https://blog.csdn.net/lly1122334/article/details/118770233

# 代码运行完成之后，在文件夹下生成 test 和  val , val change test

import splitfolders

# train:validation:test=8:1:1
splitfolders.ratio(input='/home/guoningwang/分类/classify/data', output='/home/guoningwang/分类/classify/datasets/4_categories',
                   seed=1337, ratio=(0.8, 0.2))
