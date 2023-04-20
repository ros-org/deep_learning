WTD_PROJECT = ..
include  $(WTD_PROJECT)/config.mk

# 头文件路径
INCLUDES =  -I./ \
			-I$(OPENCV_INC_DIR) \
			-I$(WTD_PROJECT)/include \
			-I$(WTD_PROJECT)/lib/librknn_api/include \
			-I$(WTD_PROJECT)/src/ENN/ \
			-I$(WTD_PROJECT)/src/process/ \
			-I$(WTD_PROJECT)/src/common/ \
			-I./include/

# lib文件路径,切记要将所有需要用到的lib的路径都进行指定(库路径，包括.so动态库)
LIB_DIR = -L$(WTD_PROJECT)/lib/librknn_api/lib -L$(WTD_PROJECT)/build/lib -L$(WTD_PROJECT)/BLL_TEST/lib
# lib文件名，注意：指定生成的lib文件（a结尾的文件）名字需要以lib开头，但是在这个位置用-l代替lib（如libBLL_TEST.a替换为-lBLL_TEST即可）。
#                提供的库可以是.a格式，也可以是.so，当是.so库时，在此处写法同.a库(如dhconfigsdk.so就写成-ldhconfigsdk即可，前面加-l并将.so去掉)。
LIB_FLAGS = -lProcess -lrknn_api -lENN -linterface -lcommon -lBLL_TEST -ldhnetsdk -ldhconfigsdk# 注意：从前往后依赖,lProcess依赖了后面的库,lrknn_api只能依赖其后面的库

SRCS     = $(wildcard *.cpp)                 # 所有cpp文件
OBJS     = $(patsubst %.cpp, %.o, $(SRCS))   # 把所有.cpp替换为.o文件，如xxx.cpp替换为xxx.o
# 将fileName.cpp生成可执行文件fileName,这行一次可以生成多个可执行文件，根据需求选择生成的目标(/BLL_TEST文件夹下所有的.cpp文件都会生成一个同名的可执行文件)
TARGETS  = $(SRCS:%.cpp=%)                   

all:$(TARGETS)

$(TARGETS): %: %.o
    # 调用./src下面的Makefile
	make -C ./src  
    # 编译命令：编译器路径 -O 目标 所有源文件 -L 所有库路径 -I 头文件路径 -l库选项   
    # 注意：-L$(OPENCV_LIB_DIR)是mk文件中指定好的opencv的lib的路径，$(OPENCV_LIBS)是在mk文件指定好的一大堆-l库名，上面已经在INCLUDES包含了opencv的头文件;这种方式和直接用上面的
    #      INCLUDES/LIB_DIR/LIB_FLAGS来指定第三方库的效果是一样的;
    # 注意：CPP(交叉编译器-g++)/CC(交叉编译器-gcc)编译器是将代码编译为可执行文件,AR ($(CROSS_COMPILER)ar)编译器是将代码编译为库;
	$(CPP) -o $@ $< $(CPP_FLAGS) $(LIB_DIR)  -L$(OPENCV_LIB_DIR) $(OPENCV_LIBS) $(INCLUDES)   $(LIB_FLAGS)  
    # 可执行文件路径，$@代表$(TARGETS)   
	mv $@ $(WTD_PROJECT)/build/bin     

$(OBJS): %.o: %.cpp 
    # 把.cpp编译为.o
	$(CPP) -c -o $@ $< $(INCLUDES)

clean:
    # 执行./src下的makefile中的clean下面的命令
	make -C ./src  clean
	rm -rf $(OBJS) $(TARGETS)
	rm -rf *.o *.bak $(WTD_PROJECT)/build/
