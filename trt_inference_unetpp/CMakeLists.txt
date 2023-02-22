cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example)

#设置C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#------------------------------------------------------设置cuda和cudnn环境-------------------------------------------------------#
#cuda用find_package命令也是能找到的，应该是可以的
#find_package(CUDA 11.6 REQUIRED)
#include_directories(${CUDA_INCLUDE_DIRS})
#message(“CUDA_LIBRARIES:${CUDA_LIBRARIES}”)
#message(“CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}”)

#1.设置cuda头文件目录
include_directories("/usr/local/cuda-11.6/include")
#2、设置cuda库目录
set(LINK_CUDA_DIR /usr/local/cuda-11.6/lib64)
LINK_DIRECTORIES(${LINK_CUDA_DIR})
#------------------------------------------------------设置cuda和cudnn环境-------------------------------------------------------#


#-----------------------------------------------------设置TensorRT的环境-----------------------------------------------------#
#1、添加头文件
include_directories("/opt/TensorRT-8.4.3.1/include")
include_directories("/opt/TensorRT-8.4.3.1/samples/common")

#2、添加库路径
set(LINK_TRT_DIR /Users/haoran/Downloads/wfdb/lib)
LINK_DIRECTORIES(${LINK_TRT_DIR})
#2、添加需要链接的库文件目录，相当于环境变量中增加LD_LIBRARY_PATH的路径的作用；该行等同于上面两行；
LINK_DIRECTORIES("/opt/TensorRT-8.4.3.1/lib")

#2、添加库路径并将其与生成的example进行链接；注意：下面第三行可以设置多个类似于${LIBS}的变量
#set(TENSORRT_LIB_PATH “/opt/TensorRT-8.4.3.1/lib”)
#file(GLOB LIBS “${TENSORRT_LIB_PATH}/*.so”)
#target_link_libraries(example ${LIBS})    #这行需要在用add_executable命令生成exe那行之后写，为了方便理解，所以写在这里；
 
#2、添加需要链接的库文件路径，注意：这里是全路径；这种方式没测试过；
#LINK_LIBRARIES("/opt/TensorRT-8.4.3.1/lib/libnvcaffe_parser.so")
#LINK_LIBRARIES("/opt/TensorRT-8.4.3.1/lib/libnvonnxparser.so")
#LINK_LIBRARIES("/opt/TensorRT-8.4.3.1/lib/libnvcaffe_parser.so.8")
#LINK_LIBRARIES("/opt/TensorRT-8.4.3.1/lib/libnvonnxparser.so.8")
#-----------------------------------------------------设置TensorRT的环境-----------------------------------------------------#


#-----------------------------------------------------设置torchVision的环境-----------------------------------------------------#
#torchVision已经安装了，相关文件已经在local下了，所以这里不需要额外设置了
#-----------------------------------------------------设置torchVision的环境-----------------------------------------------------#


#-----------------------------------------------------设置opencv4.4环境-----------------------------------------------------#
#注意：下面两行这种设置第三方库的方式也是可以的，但是前提是该第三方库已经有完备的xxxConfig.cmake.
set(OpenCV_DIR /home/bailiangliang/opencv-440/3rdparty)    #设置OpenCVConfig.cmake文件所在路径
find_package(OpenCV REQUIRED)                              #OpenCV是OpenCVConfig.cmake文件名字的前半部分(Config.cmake前面的部分)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
#-----------------------------------------------------设置opencv4.4环境-----------------------------------------------------#


#-----------------------------------------------------设置RapidJaon环境-----------------------------------------------------#
#1、设置RapidJson的头文件路径，该库只有源码，没有so库；
include_directories("~/deep_learning_tools/rapidjson/include")
#-----------------------------------------------------设置RapidJaon环境-----------------------------------------------------#


#-----------------------------------------------------设置libtorch环境环境-----------------------------------------------------#
# 下面两行的用法就是如我注释所说，经验证确实如此(我把TorchConfig.cmake修改为TorchsConfig.cmake,然后下面两行对应进行修改，仍然可以运行成功)。
# 注意：一般调用第三方库使用cmake编译时，会首先去找对应的第三方库的*Config.cmake文件，或者*.pc文件，否则是无法正常编译通过的；
set(Torch_DIR /home/bailiangliang/deep_learning_tools/libtorch/share/cmake/Torch)    #该行设置的是TorchConfig.cmake的路径，其命名规则是区“TorchConfig.cmake”字段的前半部分(Config之前的那部分)，然后接_DIR
find_package(Torch REQUIRED)                                     #该行和上一行配合使用，命名规则同上一行
message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
#-----------------------------------------------------设置libtorch环境环境-----------------------------------------------------#


# 该语句执行完当前文件夹的CMakeLists.txt之后会执行子目录“src”中的CMakeLists.txt
#add_subdirectory(src)     #当子目录src下有源文件时就需要有CMakeLists.txt，那么自然就需要该行命令，一般用于比较复杂的工程中

#基于给定的源文件生成目标可执行文件。注意：该行必须在target_link_libraries之前，否则编译会报错；
add_executable(example main.cpp trtInferUnetpp.cpp  logger.cpp postprocessIndexImage.cpp)
# sampleReporting.cpp sampleOptions.cpp logger.cpp sampleUtils.cpp getOptions.cpp sampleEngines.cpp sampleInference.cpp


#指明连接进来的库，放在add_executable之后，官方推荐使用该接口，不推荐使用link_libraries(需要放到add_executable之前)。link_libraries要设置库的全路径(如#LINK_LIBRARIES("/usr/lib/libnvinfer.so.8"))。
target_link_libraries(example  -lpthread -lm)  #为了解决找不到pthread库的一个报错
target_link_libraries(example nvcaffe_parser.so nvonnxparser.so nvcaffe_parser.so.8 nvonnxparser.so.8 nvinfer_plugin.so nvparsers.so nvinfer_plugin.so.8 nvparsers.so.8  nvinfer.so.8)
target_link_libraries(example cusolver.so.11 cudnn_ops_infer.so cufile_rdma.so cusolverMg.so.11 cuinj64.so cublas.so.11  cudnn_cnn_infer.so curand.so cufile.so.0 cusparse.so.11  cusolverMg.so  cudnn.so  cusolver.so cudnn_adv_train.so cufftw.so.10   cusparse.so cublasLt.so.11 cupti.so cufile.so cublas.so  cudnn_ops_infer.so.8  cufile_rdma.so.1 cudnn_adv_infer.so cudnn.so.8 cudnn_adv_infer.so.8    cudnn_ops_train.so.8 cublasLt.so cudnn_cnn_train.so.8 cufft.so.10 cudart.so cudnn_ops_train.so  cudnn_adv_train.so.8  cudnn_cnn_infer.so.8 cufft.so curand.so.10 cudnn_cnn_train.so curand.so 
nvToolsExt.so.1 nvToolsExt.so dl.so.2)

#下面这两行效果一样
target_link_libraries(example ${OpenCV_LIBS})
#target_link_libraries(example opencv_core opencv_imgproc opencv_highgui opencv_calib3d opencv_videoio opencv_imgcodecs)





#注意：为了编译成功，期间报了很多错误
#    1、众所周知，在Linux下第三方库都是以lib开头的，如libMyLibrary.so，但是在target_link_libraries中链接该库名的时候需要把开头的lib去掉(不确定加上lib行不行，但是make需要的Makefile文件是需要将lib去掉的)。
#    2、报错“/lib/x86_64-linux-gnu/libdl.so.2: 无法添加符号: DSO missing from command line”。解决办法，“locate libdl.so.2”命令后发现这个文件存在于很多地方(系统目录下也有)，所以直接在target_link_libraries中添加该库即可。
#    3、报错“/lib/x86_64-linux-gnu/libpthread.so.0: 无法添加符号: DSO missing from command line”。解决办法，用target_link_libraries链接“-lpthread”，这是百度出来的；
#    4、报错“在函数‘loadEngine(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int, std::ostream&)’中：MyTensorRTApp.cpp:(.text+0xc43)：对‘sample::gLogger’未定义的引用”。开始报这个错误的时候，我一直以为是我没有指定需要的so库，后面才知道这个提示的函数的定义并不在so库中，而是官方提供了.cpp源码。这些源码在tensorRT中的sample文件夹的common文件夹下(sampleReporting.cpp sampleOptions.cpp logger.cpp sampleUtils.cpp getOptions.cpp sampleEngines.cpp sampleInference.cpp)，只需要将这些文件使用add_executable命令加入即可。
#    5、add_executable命令要放在各个第三方库环境设置之后，放在target_link_libraries命令之前，否则会报错。这种一般会报未定义的引用(就是源定义未找到)，而如果是头文件未include会报函数未声明；






