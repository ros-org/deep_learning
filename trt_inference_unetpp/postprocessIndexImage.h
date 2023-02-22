#pragma once
#include"trtInferUnetpp.h"


void drawCross(cv::Mat img, cv::Point2d point, cv::Scalar color, int size, int thickness);


int get_ObjRectCenter(cv::Mat mask_img, cv::Point &center, cv::Mat &res_img);


int get_max_long_contour(cv::Mat& srcBinary, cv::Mat& dstBinary, float &fingerLong);


void thinning_iteration(cv::Mat& img, int iter);


void thinning_image(const cv::Mat& srcImg, cv::Mat& dstImg);


std::vector<cv::Point> get_points(cv::Mat &srcImg, int value);


int get_lines_for_erode_image(cv::Mat srcImg, cv::Vec4f &_find_line);


float get_line_anlge(cv::Point start_pt, cv::Point end_pt);


float get_ObjLineAngle(cv::Mat mask_img, cv::Mat &res_img);


