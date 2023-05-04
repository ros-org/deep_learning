/*==========================================================
ģ����  ���Էָ������д����������ż�������Ե�ĽǶȣ�
�ļ���  ��segResPostProcessing.cpp
����ļ�����
����    ��Liangliang Bai (liangliang.bai@leapting.com)
��Ȩ    ��<Copyright(C) 2022- Suzhou leapting Technology Co., Ltd. All rights reserved.>
�޸ļ�¼��
��  ��        �汾     �޸���   �߶���  �޸ļ�¼

2022.08.15    1.0.0.1  ������           ���ָ���ͼ�ϵķָ�������ȡ�������ֱ������
==========================================================*/

#include "segResPostProcessing.h"
#include <iostream>

using namespace std;
/*
* ------------------------------->ʵ��������halcon��select_shape����<------------------------------- *
* Parameters:
*     input��
*         cv::Mat src������ͼ��ͼ��ĸ�ʽ��8λ��ͨ����ͼ�񣬲��ұ�����Ϊ��ֵͼ��;
*         INPUT std::vector<SelectShapeType> types:Ҫ������״������
*         INPUT SelectOperation operation������Ҫ�ص��������ͣ��롢�򣩣�
*         INPUT std::vector<double> mins����Ӧ��������Сֵ��������������types��
*	      INPUT std::vector<double> maxs����Ӧ���������ֵ��������������types��
*     output��
*         cv::Mat& dst��������ѡ�������ͼ��
* returnValue:
*     None
* ------------------------------->ʵ��������halcon��select_shape����<------------------------------- *
*/
void selectShape(INPUT cv::Mat src, 
	             INPUT std::vector<SelectShapeType> types,
	             INPUT SelectOperation operation,
	             INPUT std::vector<double> mins,
	             INPUT std::vector<double> maxs,
	             OUTPUT cv::Mat& dst)
{
	if (!(types.size() == mins.size() && mins.size() == maxs.size()))
	{
		return;
	}

	int num = types.size();
	dst = cv::Mat(src.size(), CV_8UC1, cv::Scalar(0));

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(src, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	
	int cnum = contours.size();
	std::vector<std::vector<cv::Point>> selectContours;
	for (int i = 0; i < cnum; i++)
	{
		bool isAnd = true;
		bool isOr = false;
		for (int j = 0; j < num; j++)
		{
			double mind = mins[j];
			double maxd = maxs[j];
			if (mind > maxd)
			{
				mind = maxs[j];
				maxd = mins[j];
			}
			if (types[j] == SELECT_AREA)
			{
				cv::Moments moms = cv::moments(contours[i]);
				if (moms.m00 >= mind && moms.m00 <= maxd)
				{
					isAnd &= true;
					isOr |= true;
				}
				else
				{
					isAnd &= false;
					isOr |= false;
				}
			}
			else if (types[j] == SELECT_RECTANGULARITY)
			{
				cv::Moments moms = cv::moments(contours[i]);
				cv::RotatedRect rect = cv::minAreaRect(contours[i]);
				double rectangularity = moms.m00 / rect.size.area();
				if (rectangularity >= mind && rectangularity <= maxd)
				{
					isAnd &= true;
					isOr |= true;
				}
				else
				{
					isAnd &= false;
					isOr |= false;
				}
			}
			else if (types[j] == SELECT_WIDTH)
			{
				cv::Rect rect = cv::boundingRect(contours[i]);
				if (rect.width >= mind && rect.width <= maxd)
				{
					isAnd &= true;
					isOr |= true;
				}
				else
				{
					isAnd &= false;
					isOr |= false;
				}
			}
			else if (types[j] == SELECT_HEIGHT)
			{
				cv::Rect rect = cv::boundingRect(contours[i]);
				if (rect.height >= mind && rect.height <= maxd)
				{
					isAnd &= true;
					isOr |= true;
				}
				else
				{
					isAnd &= false;
					isOr |= false;
				}
			}
			else if (types[j] == SELECT_ROW)
			{
				cv::Moments moms = cv::moments(contours[i]);
				cv::Point2f center(moms.m10 / moms.m00, moms.m01 / moms.m00);
				if (center.y >= mind && center.y <= maxd)
				{
					isAnd &= true;
					isOr |= true;
				}
				else
				{
					isAnd &= false;
					isOr |= false;
				}
			}
			else if (types[j] == SELECT_COLUMN)
			{
				cv::Moments moms = cv::moments(contours[i]);
				cv::Point2f center(moms.m10 / moms.m00, moms.m01 / moms.m00);
				if (center.x >= mind && center.x <= maxd)
				{
					isAnd &= true;
					isOr |= true;
				}
				else
				{
					isAnd &= false;
					isOr |= false;
				}
			}
			else if (types[j] == SELECT_RECT2_LEN1)
			{
				cv::RotatedRect rect = cv::minAreaRect(contours[i]);
				double len = rect.size.width;
				if (rect.size.width < rect.size.height)
					len = rect.size.height;
				if (len / 2 >= mind && len / 2 <= maxd)
				{
					isAnd &= true;
					isOr |= true;
				}
				else
				{
					isAnd &= false;
					isOr |= false;
				}
			}
			else if (types[j] == SELECT_RECT2_LEN2)
			{
				cv::RotatedRect rect = cv::minAreaRect(contours[i]);
				double len = rect.size.height;
				if (rect.size.width < rect.size.height)
					len = rect.size.width;
				if (len / 2 >= mind && len / 2 <= maxd)
				{
					isAnd &= true;
					isOr |= true;
				}
				else
				{
					isAnd &= false;
					isOr |= false;
				}
			}
			else if (types[j] == SELECT_RECT2_PHI)
			{
				cv::RotatedRect rect = cv::minAreaRect(contours[i]);
				double len = rect.angle;
				if (rect.size.width < rect.size.height)
					len += 90;
				if (rect.angle >= mind && rect.angle <= maxd)
				{
					isAnd &= true;
					isOr |= true;
				}
				else
				{
					isAnd &= false;
					isOr |= false;
				}
			}
		}
		if (isAnd && operation == SELECT_AND)
			selectContours.push_back(contours[i]);
		if (isOr && operation == SELECT_OR)
			selectContours.push_back(contours[i]);
	}
	
	cv::drawContours(dst, selectContours, -1, cv::Scalar(255), cv::FILLED);
	cv::bitwise_and(src, dst, dst);
}


/*
* ---------------------------------------->�ָ�������<---------------------------------------- *
* Parameters:
*     input��
*         cv::Mat& image������ͼ�񣬷ָ����������ͼ(����ֵ��0��1��2��...);
*     output��
*         float& angle���ż�����Ե֮��ĽǶȣ�
* returnValue:
*     LT::ErrorValue������ֵ���ͣ����󲶻�
* ---------------------------------------->�ָ�������<---------------------------------------- *
*/
LT::ErrorValue postProcessingSegRes(INPUT cv::Mat& image, OUTPUT float& angle)
{
	// �п�
	if (image.empty())
	{
		cout << "postProcessingSegRes image.empty()" << endl;
		return LT::ErrorValue::IMAGE_EMPTY;
	}

	cv::Mat thresImage;
	cv::threshold(image, thresImage, 0.5, 255, 3);

	cv::Mat selectedImage;
	std::vector<SelectShapeType> selectType;
	selectType.push_back(SelectShapeType::SELECT_AREA);
	SelectOperation operation = SelectOperation::SELECT_AND;
	std::vector<double> mins;
	mins.push_back(500);
	std::vector<double> maxs;
	maxs.push_back(10000);

	selectShape(thresImage, selectType, operation, mins, maxs, selectedImage);
	cv::Mat eroImage;
	cv::Mat eroKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));  //��ʴ�����˴�С,����̫��ͻ���ż���Ч����û��
	erode(selectedImage, eroImage, eroKernel);
	
	cv::Mat dilatedImage;
	cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));  //���;����˴�С
	cv::dilate(eroImage, dilatedImage, dilateKernel);

	cv::Mat connectImage = cv::Mat::zeros(image.size(), CV_8U);
	int num_labels = connectedComponents(dilatedImage, connectImage, 8, CV_32S);
	std::cout << "num_labels:" << num_labels << endl;
	if (3 != num_labels)  //num_labels�������3������������
	{
		// Do something
		std::cout<<"Edge num wrong.............................................................."<<std::endl;
		return LT::ErrorValue::WRONG_LABEL_NUM;
	}

	// ��ȡÿ���ߵĵ㼯�ϣ���Ծ��ȡ��Ҫ��Ȼ��̫���ˣ���ͬʱ��connectImageת����������CV_8U��ʽ����
	cv::Mat indexImage(image.size(), CV_8U); 
	std::vector<std::vector<cv::Point2f>> points(2);         //������Ƿ����ֱ�ߵ����е�
	cv::Point2f point;                                       //����ʺ������ĵ������
	for (int r = 0; r < indexImage.rows; ++r)
	{
		int intervalPointsNum = 0;
		for (int c = 0; c < indexImage.cols; ++c)
		{
			int labelValue = (int)connectImage.at<int>(r, c);
			indexImage.at<uchar>(r, c) = labelValue;
			point.x = c;
			point.y = r;
			if (labelValue > 0)
			{
				intervalPointsNum++;
				if (0 == (intervalPointsNum % 20))
				{
					points[labelValue - 1].push_back(point);
				}
			}
		}
	}

	// ֱ�����
	cv::Vec4f lines[2];                                         
	cv::fitLine(points[0], lines[0], cv::DIST_L2, 0, 0.01, 0.01);
	cv::fitLine(points[1], lines[1], cv::DIST_L2, 0, 0.01, 0.01);
	
	//�Ƕȼ���
	double k1 = lines[0][1] / lines[0][0];                     //tan(alpha1)=k1
	double k2 = lines[1][1] / lines[1][0];                     //tan(alpha2)=k2
	float tanAlpha = (k2 - k1) / (1 + k1 * k2);
	if (tanAlpha < 0)
	{
		tanAlpha = -tanAlpha;
	}
	
	angle = atan(tanAlpha) * 180.f / PI ;

	return LT::ErrorValue::SUCCESS;
}


/*
* ---------------------------------------->��ȡ�㷨�汾��<---------------------------------------- *
* Parameters:
*     input��
*         None
*     output��
*         char* versionNum����ǰ�汾�ţ�
* returnValue:
*     None
* ---------------------------------------->��ȡ�㷨�汾��<---------------------------------------- *
*/
void getAlgVersion(OUTPUT char* versionNum)
{
	char version[128] = "v1.0.0.1";
	memcpy(versionNum, version, 128);
}



