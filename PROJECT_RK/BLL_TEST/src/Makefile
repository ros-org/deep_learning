WTD_PROJECT = ../..
include  $(WTD_PROJECT)/config.mk    # config.mk记录的各种变量，要使用它，只需要include一下即可

# 生成静态库.a或者动态库.so文件
LIB_TARGET = libBLL_TEST.a           # 将同目录下的所有代码生成为libBLL_TEST.a（注意：该文件名字必须以lib开头）
SRCS    = $(wildcard *.cpp)          # 该路径下所有cpp文件
OBJS    = $(SRCS:.cpp=.o)            # 将对应的cpp替换为对应的O

# $(varName)  ：代表varName变量的值
INCLUDE = -I./ \
		-I$(WTD_PROJECT)/include \
		-I$(WTD_PROJECT)/src/ENN \
		-I$(WTD_PROJECT)/lib/librknn_api/include \
		-I$(OPENCV_INC_DIR) \
		-I../include                 # 添加对应的头文件路径
	
all: $(LIB_TARGET)                   # 执行make就会执行all后边的代码

$(LIB_TARGET): $(OBJS) 
	$(AR) -rc $(LIB_TARGET) $(OBJS) 
	mv $(LIB_TARGET) $(WTD_PROJECT)/build/lib/
	
%.o: %.cpp
	$(CPP) $(INCLUDE) $(CPP_FLAGS)  -c -fpic -Wall $<

# 注意：clean下的代码必须用tab，切记不能用4个空格
clean:
	rm -f *.o  $(LIB_TARGET) *.bak  *~ $(WTD_PROJECT)/build/lib/$(LIB_TARGET)  #
