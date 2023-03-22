WTD_PROJECT = ..
include  $(WTD_PROJECT)/config.mk

INCLUDES =  -I./ \
			-I$(OPENCV_INC_DIR) \
			-I$(WTD_PROJECT)/include \
			-I$(WTD_PROJECT)/lib/librknn_api/include \
			-I$(WTD_PROJECT)/src/ENN/ \
			-I$(WTD_PROJECT)/src/process/ \
			-I$(WTD_PROJECT)/src/common/ 

LIB_DIR = -L$(WTD_PROJECT)/lib/librknn_api/lib -L$(WTD_PROJECT)/build/lib -L$(WTD_PROJECT)/lib/daHuaCam
LIB_FLAGS = -lProcess -lrknn_api -lENN -linterface -lcommon -ldhnetsdk -ldhconfigsdk

SRCS     = $(wildcard *.cpp)
OBJS     = $(patsubst %.cpp, %.o, $(SRCS))
TARGETS  = $(SRCS:%.cpp=%)

all: $(TARGETS)

$(TARGETS): %: %.o
	$(CPP) -o $@ $< $(CPP_FLAGS) $(LIB_DIR)  -L$(OPENCV_LIB_DIR) $(OPENCV_LIBS) $(INCLUDES)   $(LIB_FLAGS)
	mv $@ $(WTD_PROJECT)/build/bin

$(OBJS): %.o: %.cpp 
	$(CPP) -c -o $@ $< $(INCLUDES)

clean:
	rm -rf $(OBJS) $(TARGETS)
	rm -rf *.o   *.bak  *~ $(WTD_PROJECT)/build/
