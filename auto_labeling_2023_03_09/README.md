**测试说明 —— 用 YOLOv8  进行自动标注**



可以使用 YOLOv8 ，对图片分割任务，进行自动标注。

测试的 4 个步骤如下：

1. 在文件夹 demo 中，有 2 个子文件夹： trained_models  和 test_new_labeling，结构如下图。

   ![image-20230309172140724](assets/image-20230309172140724.png)

2. 在 trained_models 文件夹中，放入训练好的模型 pt 文件。而在 test_new_labeling 文件夹中，创建一个同名文件夹，即上图的 x1_Adam_lr1e-03_b8_e100 文件夹。

3. 把要测试的图片，放入上图的 x1_Adam_lr1e-03_b8_e100 文件夹中。

4. 在终端中，进入 demo 文件夹，输入: python3 yolov8_pred_to_labelme_jsons.py  ，即可开始测试。

标注的效果如下图：

a. 正常效果。

b. 标注不好的情况。

未完待续 ……

