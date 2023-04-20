
OPENCV_LIB_DIR=/home/shawn/WorkSpace/RV1126/Deploy/opencv_ffmepg/opencv-4.2.0/install_x86/lib
OPENCV_INC_DIR=/home/shawn/WorkSpace/RV1126/Deploy/opencv_ffmepg/opencv-4.2.0/install_x86/include/opencv4

g++ -o undistort undistort.cpp -L${OPENCV_LIB_DIR} -I${OPENCV_INC_DIR} -Wl,-rpath,${OPENCV_LIB_DIR}: \
	  -lopencv_videoio -lopencv_calib3d -lopencv_features2d \
	 -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc \
	 -lopencv_ml -lopencv_objdetect -lopencv_photo \
	 -lopencv_stitching     -lIlmImf -llibjasper -llibjpeg-turbo \
	 -llibwebp -llibtiff -lopencv_video  -lopencv_core \
	 -lzlib -Wl,-Bdynamic -fPIC   -lm -lpthread
	 
g++ -o deploy deploy.cpp -L${OPENCV_LIB_DIR} -I${OPENCV_INC_DIR} -Wl,-rpath,${OPENCV_LIB_DIR}: \
	  -lopencv_videoio -lopencv_calib3d -lopencv_features2d \
	 -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc \
	 -lopencv_ml -lopencv_objdetect -lopencv_photo \
	 -lopencv_stitching     -lIlmImf -llibjasper -llibjpeg-turbo \
	 -llibwebp -llibtiff -lopencv_video  -lopencv_core \
	 -lzlib -Wl,-Bdynamic -fPIC   -lm -lpthread