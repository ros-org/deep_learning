WTD_PROJECT = ../..
include  $(WTD_PROJECT)/config.mk

INCLUDES =  -I./ \
			-I$(OPENCV_INC_DIR) \
			-I$(WTD_PROJECT)/include \
			-I$(WTD_PROJECT)/lib/librknn_api/include \
			-I$(WTD_PROJECT)/src/ENN/ \
			-I$(WTD_PROJECT)/src/process/ \
			-I$(WTD_PROJECT)/src/common/

LIB_DIR = -L$(WTD_PROJECT)/lib/librknn_api/lib -L$(WTD_PROJECT)/build/lib
LIB_FLAGS = 
#-lProcess -lrknn_api -lENN -linterface -lcommon


all: seg_post

seg_post: main.cpp LT_VISION_COMMON.cpp segResPostProcessing.cpp

	$(CPP) -o $@ $^ $(CPP_FLAGS) $(LIB_DIR)  -L$(OPENCV_LIB_DIR) $(OPENCV_LIBS) $(INCLUDES)   $(LIB_FLAGS)
	adb push seg_post /userdata
	mv $@ $(WTD_PROJECT)/build/bin
	


clean:
	rm -rf $(OBJS) seg_post
	rm -rf *.o   *.bak  *~ $(WTD_PROJECT)/build/*
