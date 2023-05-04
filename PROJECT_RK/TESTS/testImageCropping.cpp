#include<thread>
#include"segResPostProcessing.h"
#include<iostream>
#include"imageCropping.h"



//测试//测试自适应切图
void testAdaptiveCropImg()
{
        cv::Mat srcImage;
        srcImage = cv::imread("./testImage/1.png");
        float a[6] = { 10,20,210,120,0.995, 0 };
        std::vector<float*> detRes;
        detRes.push_back(a);
        int* ptr = nullptr;
        adaptiveImageCropping imgCrop(srcImage, detRes, 2, 720, 1280, ptr);
        imgCrop.adaptiveCropImage(0, 100, 20, 20, 720, 1280);
        cv::imwrite("./testImage/4.png", imgCrop.mDstImage);
}



int main()
{
        testAdaptiveCropImg();
        
        system("pause");
        return 0;
}
