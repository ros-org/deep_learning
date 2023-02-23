cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(camera_openvino)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

#设置C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



#-----------------------------------------------------设置opencv4.4环境-----------------------------------------------------#
#注意：下面两行这种设置第三方库的方式也是可以的，但是前提是该第三方库已经有完备的xxxConfig.cmake.
# 因为 OpenCV_DIR 已经在 .bashrc 中写入，所以注释掉下面这一行。
# set(OpenCV_DIR /home/bailiangliang/opencv-440/3rdparty)    #设置OpenCVConfig.cmake文件所在路径
# find_package(OpenCV REQUIRED)                              #OpenCV是OpenCVConfig.cmake文件名字的前半部分(Config.cmake前面的部分)

message(STATUS "OpenCV library status:")
message(STATUS "    OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
#-----------------------------------------------------设置opencv4.4环境-----------------------------------------------------#


#-----------------------------------------------------设置openvino环境------------------------------------------------------#
#注意：下面两行这种设置第三方库的方式也是可以的，但是前提是该第三方库已经有完备的xxxConfig.cmake.
# set(OpenVINO_DIR /opt/intel/openvino_2022.3.0)

# OpenVINO_DIR 应该指向 OpenVINOConfig.cmake 所在的路径，如果 ~/.bashrc 中没有设置，则在此设置如下。    
set(OpenVINO_DIR /opt/intel/openvino_2022.3.0/runtime/cmake)

# set(OpenVINO_INCLUDE_DIRS /opt/intel/openvino_2022.3.0/runtime/include)
# set(OpenVINO_LIBS /opt/intel/openvino_2022.3.0/runtime/lib/intel64)

find_package(OpenVINO REQUIRED) # only Runtime component
# find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)

message(STATUS "OpenVINO_DIR : ${OpenVINO_DIR}")
message(STATUS "OpenVINO_LIBS : ${OpenVINO_LIBS}")
message(STATUS "OpenVINO_LIBRARIES : ${OpenVINO_LIBRARIES}")
message(STATUS "OpenVINO_INCLUDE_DIRS : ${OpenVINO_INCLUDE_DIRS}")


#设置openvino头文件目录
include_directories("/opt/intel/openvino_2022.3.0/samples/cpp/common/utils/include")
include_directories("/opt/intel/openvino_2022.3.0/samples/cpp/common/format_reader")
include_directories("/opt/intel/openvino_2022.3.0/samples/cpp/common/utils/include/samples")
include_directories("/opt/intel/openvino_2022.3.0/runtime/include")

# include_directories("/home/bailiangliang/OPENVINO_TEST")
# 上面为 Liangliang 的路径，下面修改为我的路径。
include_directories("/home/leapting/jun_ws/deploy_openvino/src/camera_openvino/src")

#Set openvino liarary path
set(LINK_OPENVINO_DIR /opt/intel/openvino_2022.3.0/runtime/lib/intel64)
LINK_DIRECTORIES(${LINK_OPENVINO_DIR})
#-----------------------------------------------------设置openvino环境------------------------------------------------------#

#------------------------------------------------------设置dflags环境-------------------------------------------------------#
#include_directories("/usr/local/include/gflags")
#------------------------------------------------------设置dflags环境-------------------------------------------------------#

# ================== 开始： 从原有 camera_tensorrt 中拷贝过来的 CMakeLists.txt 设置 ==========================

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
  
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  tf
  rospy
  std_msgs
  message_generation
  dynamic_reconfigure
  #OpenCV    # 调用ros自带的opencv,直接去find opencv包
)
 
generate_dynamic_reconfigure_options(
  cfg/camera_openvino.cfg
)


catkin_package(
   CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs  message_runtime 
)

find_package(Eigen3 REQUIRED)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  #调用第三方需要设置include的路径。/opencv-440/3rdparty/OpenCVConfig.cmake文件中设置了OpenCV_INCLUDE_DIRS变量(指定好了所有include的路径)；
  #  ${OpenCV_INCLUDE_DIRS}   
)
add_definitions(${EIGEN3_DEFINITIONS})


# ================== 结束： 从原有 camera_tensorrt 中拷贝过来的 CMakeLists.txt 设置 ==========================

#基于给定的源文件生成目标可执行文件。注意：该行必须在 target_link_libraries 之前，否则编译会报错；
# 测试，下面使用 camera_openvino.cpp 和 main.cpp 相互替换，看是否可以用 catkin_make 直接编译
add_executable(camera_openvino src/camera_openvino.cpp src/openvinoInferenceUnetpp.cpp  
  src/slog.cpp src/opencv_wrapper.cpp src/yuv_nv12.cpp src/format_reader.cpp src/common.cpp 
  src/latency_metrics.cpp  src/common.hpp src/csv_dumper.hpp src/ocv_common.hpp src/console_progress.hpp 
  src/slog.hpp src/latency_metrics.hpp src/MnistUbyte.cpp src/bmp.cpp src/args_helper.cpp src/args_helper.hpp)

#指明连接进来的库，放在add_executable之后，官方推荐使用该接口，不推荐使用link_libraries(需要放到add_executable之前)。link_libraries要设置库的全路径(如#LINK_LIBRARIES("/usr/lib/libnvinfer.so.8"))。
target_link_libraries(camera_openvino  -lpthread -lm)  #为了解决找不到pthread库的一个报错

# 链接 ROS 动态库。
target_link_libraries(camera_openvino 
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  )
#set opencv library link,下面这两行效果一样
#target_link_libraries(camera_openvino opencv_core opencv_imgproc opencv_highgui opencv_calib3d opencv_videoio opencv_imgcodecs)
target_link_libraries(camera_openvino openvino::runtime)
# target_link_libraries(camera_openvino ${OpenCV_LIBS} openvino::runtime)
# 下面增加 OpenVINO 动态库的部分，似乎只要在上一行加上 openvino::runtime 即可。
#Set openvino library link 
# target_link_libraries(camera_openvino openvino_c openvino.so.2230  openvino_c.so.2230 openvino_c.so openvino_tensorflow_frontend.so.2230 openvino_tensorflow_frontend.so openvino.so
# openvino_onnx_frontend.so.2230  openvino_paddle_frontend.so openvino_onnx_frontend.so openvino_paddle_frontend.so.2230)

#Set gflags library link
target_link_libraries(camera_openvino gflags)


#注意：为了编译成功，期间报了很多错误
#    1、众所周知，在Linux下第三方库都是以lib开头的，如libMyLibrary.so，但是在target_link_libraries中链接该库名的时候需要把开头的lib去掉(不确定加上lib行不行，但是make需要的Makefile文件是需要将lib去掉的)。
#    2、报错“/lib/x86_64-linux-gnu/libdl.so.2: 无法添加符号: DSO missing from command line”。解决办法，“locate libdl.so.2”命令后发现这个文件存在于很多地方(系统目录下也有)，所以直接在target_link_libraries中添加该库即可。
#    3、报错“/lib/x86_64-linux-gnu/libpthread.so.0: 无法添加符号: DSO missing from command line”。解决办法，用target_link_libraries链接“-lpthread”，这是百度出来的；
#    4、报错“在函数‘loadEngine(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int, std::ostream&)’中：MyTensorRTApp.cpp:(.text+0xc43)：对‘sample::gLogger’未定义的引用”。开始报这个错误的时候，我一直以为是我没有指定需要的so库，后面才知道这个提示的函数的定义并不在so库中，而是官方提供了.cpp源码。这些源码在tensorRT中的sample文件夹的common文件夹下(sampleReporting.cpp sampleOptions.cpp logger.cpp sampleUtils.cpp getOptions.cpp sampleEngines.cpp sampleInference.cpp)，只需要将这些文件使用add_executable命令加入即可。
#    5、add_executable命令要放在各个第三方库环境设置之后，放在target_link_libraries命令之前，否则会报错。这种一般会报未定义的引用(就是源定义未找到)，而如果是头文件未include会报函数未声明；






