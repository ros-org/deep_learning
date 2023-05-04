WTD_PROJECT = ..
include  $(WTD_PROJECT)/config.mk

INCLUDES = -I./ \
		-I$(OPENCV_INC_DIR) \
		-I$(WTD_PROJECT)/include \
        -I$(WTD_PROJECT)/lib/librknn_api/include \
		-I$(WTD_PROJECT)/src/ENN/

LIB_DIR = -L$(WTD_PROJECT)/lib/librknn_api/lib -L$(WTD_PROJECT)/build/lib
LIB_FLAGS = -lrknn_api -lENN -lcommon

SRCS     = $(wildcard *.cpp)
OBJS     = $(patsubst %.cpp, %.o, $(SRCS))
TARGETS  = $(SRCS:%.cpp=%)
$(info $(OBJS))
$(info $(TARGETS))

all: $(TARGETS)

$(TARGETS): %: %.o
	$(CPP) -o $@ $< $(CPP_FLAGS) $(LIB_DIR)  -L$(OPENCV_LIB_DIR) $(OPENCV_LIBS) $(INCLUDES)   $(LIB_FLAGS)
	mv $@ $(WTD_PROJECT)/build

$(OBJS): %.o: %.cpp 
	$(CPP) -c -o $@ $< $(INCLUDES)

# all: main test_opencv test_yolov5 test_configer

# main:main.cpp
# 	$(CPP) -o $@ $^ $(CPP_FLAGS) $(LIB_DIR) $(LIB_FLAGS) -L$(OPENCV_LIB_DIR) $(OPENCV_LIBS) $(INCLUDES) 
# 	mv $@ $(WTD_PROJECT)/build

# test_opencv: test_opencv.cpp
# 	$(CPP) -o $@ $^ $(CPP_FLAGS) -L$(OPENCV_LIB_DIR) $(OPENCV_LIBS) $(INCLUDES) 
# 	mv $@ $(WTD_PROJECT)/build

# test_yolov5:test_yolov5.cpp
# 	$(CPP) -o $@ $^ $(CPP_FLAGS) $(LIB_DIR) $(LIB_FLAGS) -L$(OPENCV_LIB_DIR) $(OPENCV_LIBS) $(INCLUDES) 
# 	mv $@ $(WTD_PROJECT)/build

# test_configer:test_configer.cpp
# 	$(CPP) -o $@ $^ $(CPP_FLAGS) $(LIB_DIR) $(LIB_FLAGS) -L$(OPENCV_LIB_DIR) $(OPENCV_LIBS) $(INCLUDES) 
# 	mv $@ $(WTD_PROJECT)/build

clean:
	rm -rf $(OBJS) $(TARGETS)
	rm -rf *.o   *.bak  *~ $(WTD_PROJECT)/build/
