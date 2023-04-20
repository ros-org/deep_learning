#include<iostream>
#include <ctime> 
#include<opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace cv;
using namespace std;

int test_undistort_points();
int test_cap();

int main()
{
	test_undistort_points();
	return 0;
}

int test_undistort_points()
{
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	float cam_inner_param[] = {
		1675.487315277889, 0, 1187.092837811729,
		0, 1678.457891873492, 795.7431556734941,
		0, 0, 1
	};
	cameraMatrix.at<double>(0,0) = cam_inner_param[0];
	cameraMatrix.at<double>(0,1) = cam_inner_param[1];
	cameraMatrix.at<double>(0,2) = cam_inner_param[2];
	cameraMatrix.at<double>(1,1) = cam_inner_param[4];
	cameraMatrix.at<double>(1,2) = cam_inner_param[5];
 
	Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
	float dist_coeffs_param[] = {
		-0.3334986259058141, 0.02496279995241963, -0.006675057545274884, -0.005341053524774705, 0.06517534336438081
	};
	distCoeffs.at<double>(0,0) = dist_coeffs_param[0];
	distCoeffs.at<double>(1,0) = dist_coeffs_param[1];
	distCoeffs.at<double>(2,0) = dist_coeffs_param[2];
	distCoeffs.at<double>(3,0) = dist_coeffs_param[3];
	distCoeffs.at<double>(4,0) = dist_coeffs_param[4];

	Mat image = imread("images/11.jpg");
	int BOARDSIZE[2]{ 7,9 };//棋盘格每行每列角点个数
	Mat img_gray;
	cvtColor(image, img_gray, COLOR_BGR2GRAY);
	//检测角点
	vector<Point2f> img_corner_points;//保存每张图检测到的角点
	bool found_success = findChessboardCorners(img_gray, Size(BOARDSIZE[0], BOARDSIZE[1]),
		img_corner_points,
		CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

	if (found_success){
		//迭代终止条件
		TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);

		//进一步提取亚像素角点
		cornerSubPix(img_gray, img_corner_points, Size(11, 11),
			Size(-1, -1), criteria);

		//绘制角点
		// drawChessboardCorners(image, Size(BOARDSIZE[0], BOARDSIZE[1]), img_corner_points,
		// 	found_success);

		// objpoints_img.push_back(obj_world_pts);//从世界坐标系到相机坐标系
		// images_points.push_back(img_corner_points);
	}
	std::vector<cv::Point2f> outputUndistortedPoints;
	Mat dst;
	undistort(image, dst, cameraMatrix, distCoeffs);
	undistortPoints(img_corner_points, outputUndistortedPoints, cameraMatrix, distCoeffs, cv::noArray(), cameraMatrix); 
	drawChessboardCorners(dst, Size(BOARDSIZE[0], BOARDSIZE[1]), outputUndistortedPoints,found_success);
	imwrite("res.jpg", dst);
    return 0;
}

int test_cap()
{
	VideoCapture inputVideo(0);
	if(!inputVideo.isOpened()){
		std::cout << "video is not opened\n\n"<<endl;
	}
	else{
		std::cout << "video is opened \n\n"<<endl;
	}
//  Matlab 标定的相机参数
	Mat frame, frameCalibration;
	inputVideo >> frame;
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	float cam_inner_param[] = {
		1675.487315277889, 0, 1187.092837811729,
		0, 1678.457891873492, 795.7431556734941,
		0, 0, 1
	};
	cameraMatrix.at<double>(0,0) = cam_inner_param[0];
	cameraMatrix.at<double>(0,1) = cam_inner_param[1];
	cameraMatrix.at<double>(0,2) = cam_inner_param[2];
	cameraMatrix.at<double>(1,1) = cam_inner_param[4];
	cameraMatrix.at<double>(1,2) = cam_inner_param[5];
 
	Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
	float dist_coeffs_param[] = {
		-0.3334986259058141, 0.02496279995241963, -0.006675057545274884, -0.005341053524774705, 0.06517534336438081
	};
	distCoeffs.at<double>(0,0) = dist_coeffs_param[0];
	distCoeffs.at<double>(1,0) =  dist_coeffs_param[1];
	distCoeffs.at<double>(2,0) =  dist_coeffs_param[2];
	distCoeffs.at<double>(3,0) = dist_coeffs_param[3];
	distCoeffs.at<double>(4,0) = dist_coeffs_param[4];
 
	Mat view, rview, map1, map2;
	Size image_Size;
	image_Size = frame.size();
	
	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), cameraMatrix, image_Size, CV_16SC2, map1, map2);
	// initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, image_Size, 0.5, image_Size, 0),image_Size, CV_16SC2, map1, map2);
 
	while(1){
		inputVideo >> frame;
		if(frame.empty()) {
			break;
		}
		remap(frame, frameCalibration, map1, map2, INTER_LINEAR);
		imshow("Original_image",frame);
		imshow("Calibrated_image", frameCalibration);
		char key = waitKey(1);
		if(key == 27 || key == 'q' || key == 'Q') {
			break;
		}
	
	return 0;
}
