WTD_PROJECT = ../..
include  $(WTD_PROJECT)/config.mk

LIB_TARGET = libcommon.a
SRCS    = $(wildcard *.cpp)
OBJS    = $(SRCS:.cpp=.o)

INCLUDE = -I./ \
		  -I$(WTD_PROJECT)/include
        
all: $(LIB_TARGET)

$(LIB_TARGET): $(OBJS)
	$(AR) -rc $(LIB_TARGET) $(OBJS) 
	mv $(LIB_TARGET) $(WTD_PROJECT)/build/lib/
	
%.o: %.cpp
	$(CPP) $(INCLUDE) $(CFLAGS)  -c -fpic -Wall $<

clean:
	rm -f *.o  *.a *.bak  *~ $(WTD_PROJECT)/build/lib/$(LIB_TARGET)
