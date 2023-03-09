// #include"uvc_cameras.h"
// #include"lineFitting.h"
// #include<opencv.hpp>
// #include"3D_reconstruction.h"
// #include"TP_LiNK_camera.h"
// #include"call_tplink_using_Hik.h"
#include<thread>
#include"segResPostProcessing.h"

using namespace std;
using namespace cv;


int main()
{
	char versionNum[128];
	memset(versionNum, 0, 128);
	getAlgVersion(versionNum);
	std::cout << "versionNum:" << versionNum << std::endl;


	float angle = -999;
	cv::Mat image = imread("index_image_NG_4.bmp", cv::IMREAD_GRAYSCALE);
	cout << "image.rows" << image.rows << endl;
	cout << "image.cols" << image.cols << endl;
	postProcessingSegRes(image, angle);

	cout << "angle:" << angle << endl;

	// system("pause");
	return 0;



}





