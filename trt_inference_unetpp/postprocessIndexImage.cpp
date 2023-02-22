#include"postprocessIndexImage.h"


void drawCross(cv::Mat img, cv::Point2d point, cv::Scalar color, int size, int thickness)
{
   cv::line(img, cv::Point2d(point.x - size / 2, point.y), cv::Point2d(point.x + size / 2, point.y), color, thickness, cv::LINE_AA, 0);
   cv::line(img, cv::Point2d(point.x, point.y - size / 2), cv::Point2d(point.x, point.y + size / 2), color, thickness, cv::LINE_AA, 0);
   return;
 }

int get_ObjRectCenter(cv::Mat mask_img, cv::Point &center, cv::Mat &res_img) 
{
	res_img = mask_img.clone();
	cv::Mat gray_Img;
	cvtColor(mask_img, gray_Img, cv::COLOR_BGR2GRAY);
	//cv::imshow("mask", mask_img);

	cv::Mat mask_thr_circle;
	threshold(gray_Img, mask_thr_circle, 50, 255, cv::THRESH_BINARY);
	cv::imshow("mask_thr_circle", mask_thr_circle);
	
	cv::Mat open_result;
	cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
	morphologyEx(mask_thr_circle, open_result, cv::MORPH_OPEN, element);
	morphologyEx(open_result, open_result, cv::MORPH_CLOSE, element);
	
	std::vector<cv::Vec4i> hierarchy;
	std::vector<std::vector<cv::Point>> contours;
	findContours(open_result, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());
	if (contours.size() < 1) 
	{
		std::cout << "contours size is : " << contours.size() << std::endl;
		return -1;
	}

	// 获取最大两个轮廓
	std::vector<int> index_list;
	std::vector<double> area_list;
	for (int i = 0; i < contours.size(); ++i) 
	{
		cv::Rect rect = cv::boundingRect(contours[i]);
		if ((rect.width > int(mask_img.cols * 0.05)) && (rect.height > int(mask_img.rows * 0.05))) 
		{
			index_list.push_back(i);
			area_list.push_back(contourArea(contours[i]));
			std::cout << "area: " << contourArea(contours[i]) << std::endl;
		}
	}
	
	if (area_list.size() < 1)
	{
		std::cout << "area_list size is : " << area_list.size() << std::endl;
		return -1;
	}
	

	std::vector<int> index_list_temp;
	std::vector<double> area_list_temp = area_list;
	sort(area_list_temp.begin(), area_list_temp.end());
	for (int i = 0; i < 2; i++)
	{ 
		for (int j = 0; j < area_list.size(); j++)
		{
			if (area_list_temp[i] == area_list[j]) 
			{
				index_list_temp.push_back(j);
			}
		}
	}

	std::vector<cv::Point> obj_cen_list;
	for (int i = 0; i < index_list_temp.size(); i++)
	{
		drawContours(res_img, contours, index_list_temp[i], cv::Scalar(0, 0, 255), 1, 8);
		cv::RotatedRect minRect = minAreaRect(contours[index_list_temp[i]]);
		
		cv::Point2f P[4];
		minRect.points(P);
		for (int j = 0; j <= 3; j++)
		{
			line(res_img, P[j], P[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
		}
		obj_cen_list.push_back(minRect.center);
		drawCross(res_img, minRect.center, (0, 0, 255), 15, 2);
	}

	line(res_img, obj_cen_list[0], obj_cen_list[1], cv::Scalar(255, 255, 0), 2);
	
	center.x = int((obj_cen_list[0].x + obj_cen_list[1].x) / 2.0);
	center.y = int((obj_cen_list[0].y + obj_cen_list[1].y) / 2.0);
	cv::circle(res_img, center, 5, cv::Scalar(0, 0, 255), -1, 8);

	cv::imshow("res_img", res_img);
    //cv::imshow("open_result", open_result);
	cv::waitKey(0);
}



int get_max_long_contour(cv::Mat& srcBinary, cv::Mat& dstBinary, float &fingerLong)
{
	int contours_index = -1;
	float minRect_width = 0;
	float minRect_height = 0;
	float maxContour = 0;
	cv::Mat contours_Image(srcBinary.rows, srcBinary.cols, CV_8UC1, cv::Scalar::all(0));

	std::vector<cv::Vec4i> hierarchy;
	std::vector<std::vector<cv::Point>> contours;
	findContours(srcBinary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());
	for (int i = 0; i < contours.size(); i++) 
	{
		cv::RotatedRect minRect = minAreaRect(contours[i]);;
		minRect_width = minRect.size.width;
		minRect_height = minRect.size.height;
		if (minRect_width > minRect_height) 
		{
			minRect_height = minRect.size.width;
			minRect_width = minRect.size.height;
		}
		if (minRect_height > maxContour) 
		{
			maxContour = minRect_height;
			fingerLong = minRect_height;
			contours_index = i;
		}
	}
	if (contours_index != -1) 
	{
		drawContours(contours_Image, contours, contours_index, cv::Scalar::all(255), -1, 8);
		dstBinary = contours_Image.clone();
		return 1;
	}
	else 
	{
		return -1;
	}
}


void thinning_iteration(cv::Mat& img, int iter)
{
	CV_Assert(img.channels() == 1);
	CV_Assert(img.depth() != sizeof(uchar));
	CV_Assert(img.rows > 3 && img.cols > 3);
	cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);
	int nRows = img.rows;
	int nCols = img.cols;
	if (img.isContinuous()) 
	{
		nCols *= nRows;
		nRows = 1;
	}
	int x, y;
	uchar *pAbove;
	uchar *pCurr;
	uchar *pBelow;
	uchar *nw, *no, *ne;
	uchar *we, *me, *ea;
	uchar *sw, *so, *se;
	uchar *pDst;
	pAbove = NULL;
	pCurr = img.ptr<uchar>(0);
	pBelow = img.ptr<uchar>(1);
	for (y = 1; y < img.rows - 1; ++y) 
	{
		pAbove = pCurr;
		pCurr = pBelow;
		pBelow = img.ptr<uchar>(y + 1);
		pDst = marker.ptr<uchar>(y);
		no = &(pAbove[0]);
		ne = &(pAbove[1]);
		me = &(pCurr[0]);
		ea = &(pCurr[1]);
		so = &(pBelow[0]);
		se = &(pBelow[1]);
		for (x = 1; x < img.cols - 1; ++x) 
		{
			nw = no;
			no = ne;
			ne = &(pAbove[x + 1]);
			we = me;
			me = ea;
			ea = &(pCurr[x + 1]);
			sw = so;
			so = se;
			se = &(pBelow[x + 1]);
			int A = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
				(*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
				(*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
				(*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
			int B = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
			int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
			int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);
			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
				pDst[x] = 1;
		}
	}
	img &= ~marker;

	return;
}


void thinning_image(const cv::Mat& srcImg, cv::Mat& dstImg)
{
	dstImg = srcImg.clone();
	dstImg /= 255;
	cv::Mat prev = cv::Mat::zeros(dstImg.size(), CV_8UC1);
	cv::Mat diff;
	do {
		thinning_iteration(dstImg, 0);
		thinning_iteration(dstImg, 1);
		absdiff(dstImg, prev, diff);
		dstImg.copyTo(prev);
	} while (countNonZero(diff) > 0);
	dstImg *= 255;

	return;
}


std::vector<cv::Point> get_points(cv::Mat &srcImg, int value)
{
	int nl = srcImg.rows;
	int nc = srcImg.cols * srcImg.channels();
	std::vector<cv::Point> points;
	for (int j = 0; j < nl; j++) 
	{
		uchar* data = srcImg.ptr<uchar>(j);
		for (int i = 0; i < nc; i++) 
		{
			if (data[i] == value) 
			{
				points.push_back(cv::Point(i, j));
			}
		}
	}

	return points;
}


int get_lines_for_erode_image(cv::Mat srcImg, cv::Vec4f &_find_line)
{
	cv::Mat src_Img = srcImg.clone();

	cv::Mat thin_Img;
	thinning_image(src_Img, thin_Img);
	std::vector<cv::Point> points = get_points(thin_Img, 255);
	if (points.size() < 20)
		return -1;

	cv::Vec4f line_point;
	fitLine(points, line_point, cv::DIST_HUBER, 0, 0.01, 0.01);
	double cos_theta = line_point[0];
	double sin_theta = line_point[1];
	cv::Point2f pt1, pt2;
	double liner_centure_x = line_point[2], liner_centure_y = line_point[3];
	double phi = atan2(sin_theta, cos_theta) + CV_PI / 2.0;
	double rho = liner_centure_y * cos_theta - liner_centure_x * sin_theta;
	if (phi < CV_PI / 4. || phi > 3.* CV_PI / 4.) 
	{
		pt1 = cv::Point2f(rho / cos(phi), 0);
		pt2 = cv::Point2f((rho - src_Img.rows * 1. * sin(phi)) / cos(phi), src_Img.rows * 1.);
	}
	else 
	{
		pt1 = cv::Point2f(0, rho / sin(phi));
		pt2 = cv::Point2f(src_Img.cols * 1., (rho - src_Img.cols * 1. * cos(phi)) / sin(phi));
	}

	_find_line = cv::Vec4f(pt1.x, pt1.y, pt2.x, pt2.y);
	line(thin_Img, pt1, pt2, cv::Scalar(255), 2);

	return 1;
}


float get_line_anlge2(cv::Point start_pt, cv::Point end_pt)
{
	if (start_pt.x == end_pt.x) 
	{
		start_pt.x += 1;
	}
	if (start_pt.y == end_pt.y) 
	{
		start_pt.y += 1;
	}

	int line_y = (start_pt.y - end_pt.y);
	int line_x = (start_pt.x - end_pt.x);
	float line_K = (line_y * 1.0) / (line_x * 1.0);
	float line_angle = atan(line_K) * 180 / CV_PI;

	if (start_pt.y > end_pt.y) 
	{
		if (line_angle > 0 && line_angle <= 90)
			line_angle += 90;
		else
			line_angle += 270;
	}
	else 
	{
		if (line_angle < 0 && line_angle >= -90)
			line_angle += 90;
		else
			line_angle += 270;
	}
	return line_angle;
}


float get_ObjLineAngle(cv::Mat mask_img, cv::Mat &res_img) 
{
	res_img = mask_img.clone();
	cv::Mat gray_Img;
	cvtColor(mask_img, gray_Img, cv::COLOR_BGR2GRAY);
	//cv::imshow("mask", mask_img);

	cv::Mat mask_thr_circle;
	threshold(gray_Img, mask_thr_circle, 50, 255, cv::THRESH_BINARY);
	//cv::imshow("mask_thr_circle", mask_thr_circle);
	//cv::waitKey(0);

	cv::Mat open_result;
	cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	morphologyEx(mask_thr_circle, open_result, cv::MORPH_OPEN, element);
	morphologyEx(open_result, open_result, cv::MORPH_CLOSE, element);

	cv::Mat mask_Img;
	float finger_long = 0;
	if (get_max_long_contour(open_result, mask_Img, finger_long) != 1) 
	{
		std::cout << "->no detection mask." << std::endl;
		return -1;
	}
	cv::imshow("res_mask", mask_Img);


	cv::Vec4f find_line;
	cv::Point line_point_1, line_point_2;
	if (get_lines_for_erode_image(mask_Img, find_line) == -1) 
	{
		return -1;
	}
	else 
	{
		line_point_1 = cv::Point(int(find_line[0]), int(find_line[1]));
		line_point_2 = cv::Point(int(find_line[2]), int(find_line[3]));
	}
	cv::line(res_img, line_point_1, line_point_2, cv::Scalar(0, 255), 2);
	float anlge = get_line_anlge2(line_point_1, line_point_2);
	std::string str_anlge = cv::format("anlge: %.5f", anlge);
	putText(res_img, str_anlge, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8, false);

	cv::imshow("res_img", res_img);
	cv::waitKey(0);
	return anlge;
}

/*
int main() 
{
	std::vector<cv::String> filenames; 
	cv::String folder =  "./test_img/rect/";
	cv::glob(folder, filenames); 
	for (size_t i = 0; i < filenames.size(); ++i)
	{
		std::cout << filenames[i] << std::endl;
		cv::Mat src = cv::imread(filenames[i]);
		
		// 获取两个矩形
		cv::Point cen;
	    cv::Mat display;
	    get_ObjRectCenter(src, cen, display);

		// 获取直线
		//cv::Mat display;
		//float line_angle = get_ObjLineAngle(src, display);
	}
}
*/
