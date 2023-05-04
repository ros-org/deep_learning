##############################################
# MurphyPlus   ARM     
##############################################
PLATFORM = ARM

# PROJECT_DIR_ = .

ifeq ($(P),)    # 如果$(P)为空
	PRODUCT = MY_PROJECT
endif

ifeq ($(PLATFORM),ARM)    # # 如果$(PLATFORM)==ARM
NFS_PATH       =
# LIB_PATH       = $(PROJECT_DIR_)/build/lib
# INCLUDE_PATH   = $(PROJECT_DIR_)/include
CROSS_COMPILER = arm-linux-gnueabihf-
endif

######################################################################
# Automatically generated configuration file, please don't edit
######################################################################

 
AS = $(CROSS_COMPILER)as
 
LD = $(CROSS_COMPILER)ld
 
CC = $(CROSS_COMPILER)gcc
 
CPP= $(CROSS_COMPILER)g++
 
AR = $(CROSS_COMPILER)ar
 
NM = $(CROSS_COMPILER)nm

RANLIB = $(CROSS_COMPILER)ranlib
 
STRIP = $(CROSS_COMPILER)strip    # 将可执行文件变得更小的 编译器
 
OBJDUMP = $(CROSS_COMPILER)objdump

CP   = cp
RM   = rm
TYPE = release
MAKE = make

OPENCV_LIB_DIR = $(WTD_PROJECT)/lib/opencv/lib
OPENCV_INC_DIR = $(WTD_PROJECT)/lib/opencv/include

OPENCV_LIBS = -Wl,-rpath,$(OPENCV_LIB_DIR): -lopencv_videoio -lopencv_calib3d -lopencv_features2d \
	 -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc \
	 -lopencv_ml -lopencv_objdetect -lopencv_photo \
	 -lopencv_stitching     -lIlmImf -llibjasper -llibjpeg-turbo \
	 -llibwebp -llibtiff -lopencv_video  -lopencv_core \
	 -lzlib -Wl,-Bdynamic -fPIC  -lavformat -lavcodec -lavdevice -lavfilter \
	 -lavutil -lpostproc -lswresample \
	 -lswscale -lx264 -lm -lpthread 

CFLAGS = -std=c99 -lm -O2
CPP_FLAGS = 

