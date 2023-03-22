make clean
make

# 注意：使用echo $PATH命令检查交叉编译器的路径是否在PATH中；
# -------------------------------------------》将第三方库push到开发板《------------------------------------------- #
#adb push build/test_opencv /userdata
#adb push build/test_yolov5 /userdata
adb push lib/daHuaCam/*.so /userdata/
adb shell "mkdir -p /userdata/lib"                                                 # 在 userdata/ 下创建 lib 文件夹，                                                                                  # 将大dahua提供的SDK库push到开发板userdata下面
adb shell "mv -f /userdata/libdhconfigsdk.so /userdata/lib/libdhconfigsdk.so"     # 将so文件从userdata移动到lib下，因为直接移动到lib会失败
adb shell "mv -f /userdata/libdhnetsdk.so /userdata/lib/libdhnetsdk.so"                 # 将so文件从userdata移动到lib下，因为直接移动到lib会失败
adb push lib/daHuaCam/*.so /userdata/lib/                                                                            # 将大dahua提供的SDK库push到开发板,直接推过去可能会失败
adb push lib/librknn_api/lib64/*.so /userdata/lib/                                                               # 将大rknn提供的SDK库push到开发板
adb push lib/opencv/lib/*.so* /userdata/lib/                                                                           # 将交叉编译的opencv库放到开发板/userdata/lib下
#注意：上面将那么多第三方库推到了userdata/lib下，所以需要将该路径添加到环境变量中才可以正常被调用
# -------------------------------------------》将第三方库push到开发板《------------------------------------------- #


# --------------------------------------》将编译生成的可执行文件push到开发板《-------------------------------------- #
adb push build/bin/* /userdata                                                   # 将编译生成的可执行文件push到开发板上
# --------------------------------------》将编译生成的可执行文件push到开发板《-------------------------------------- #


# ------------------------------------》创建一个文件夹并将模型文件push到开发板《------------------------------------- #
adb shell "mkdir -p /userdata/output"                                             # output文件夹是用来存放日志相关的文件的
adb shell "mkdir -p /userdata/models"                                           # -p是防止该文件夹已存在
adb push models/resnet.rknn /userdata/models/                                   # 将天气分类模型push到models文件夹下
adb push models/resnet-bll.rknn /userdata/models/                               # 将清洁度分类模型push到models文件夹下
adb push models/yolov5s.rknn /userdata/models/                                  # 将检测模型push到models文件夹下
adb push models/unetpp.rknn /userdata/models/                                   # 将分割模型push到models文件夹下
adb shell "mkdir -p /userdata/data"                                             # 在开发板上创建data文件夹
adb push data/*.jpg /userdata/data/                                           # 在data文件夹中push一张测试图用于离线测试
# adb push data/test_0.jpg /userdata/data/                                                     # 在data文件夹中push一张测试图用于离线测试
# adb push data/test_1.jpg /userdata/data/                                                     # 在data文件夹中push一张测试图用于离线测试
# ------------------------------------》创建一个文件夹并将模型文件push到开发板《------------------------------------- #


# ------------------------------》删除output下的文件，添加环境变量并执行某一个可执行文件《------------------------------ #
# 注意：在板子端执行，需要在命令行前加"adb shell“;下面第二行是两个命令，中间用&&符号（先添加环境变量再运行可执行文件）;使用命令export LD_LIBRARY_PATH=/userdata/lib将/userdata/lib
#      加入到环境变量后，我们将so库push到这个路径，可执行文件在运行时就会到这个路径去找它依赖的库内容;
adb shell "rm -rf /userdata/output/*"
adb shell "export LD_LIBRARY_PATH=/userdata/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}&& /userdata/main"
# adb shell "export LD_LIBRARY_PATH=/userdata/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}&& /userdata/bll_main"
# adb shell "export LD_LIBRARY_PATH=/userdata/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}&& /userdata/test_yolov5"
# adb shell "export LD_LIBRARY_PATH=/userdata/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}&& /userdata/test_seg"
# adb shell "export LD_LIBRARY_PATH=/userdata/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}&& /userdata/test_opencv"
# adb shell "export LD_LIBRARY_PATH=/userdata/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}&& /userdata/test_seg /userdata/data/test_seg.jpg"
# ------------------------------》删除output下的文件，添加环境变量并执行某一个可执行文件《------------------------------ #






