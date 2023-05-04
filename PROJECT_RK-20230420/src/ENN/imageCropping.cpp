/*==========================================================
模块名  ：自适应切图程序；
裁剪方法：输入原图+检测模型推理结果->将指定类别的框坐标进行上采样，扩大几个像素->判断是否到边缘->切图
          ->计算自适应填充像素个数(宽高比与下个模型推理尺寸宽高比保持一致，防止resize导致图像扭曲)->自适应填充像素生成新的图
文件名  ：imageCropping.cpp
相关文件：无
作者    ：Liangliang Bai (liangliang.bai@leapting.com)
版权    ：<Copyright(C) 2022- Huzhou leapting Technology Co., Ltd. All rights reserved.>
修改记录：
日  期        版本     修改人   走读人
2023.04.25    1.0.0.1  白亮亮
==========================================================*/

#include"imageCropping.h"


/*
* -------------------------------------------->默认构造<-------------------------------------------- *
* 显示地定义默认构造，我在下面定义了拷贝构造和赋值构造。
* Parameters:
*     input：
*         None
*     output：
*         None
* returnValue:
*     None
* -------------------------------------------->默认构造<-------------------------------------------- *
*/
//adaptiveImageCropping::adaptiveImageCropping() {}



/*
* -------------------------------------------->有参构造<-------------------------------------------- *
* Parameters:
*     input：
*         srcImage:上一次推理所使用的原图(未经过resize等任何操作)
*         dstImage：被操作后的图像(下一个模型要推理的图像)
*         detRes：上一次检测模型的推理结果
*         clsNum：上次检测的类别数
*         infeHeight：上次推理输入网络的图像高(已经过resize等前处理操作)
*         infeWidth：上次推理输入网络的图像宽(已经过resize等前处理操作)
*         ptr：备用指针
*     output：
*         None
* returnValue:
*     None
* -------------------------------------------->有参构造<-------------------------------------------- *
*/
adaptiveImageCropping::adaptiveImageCropping(IN const cv::Mat & srcImage, IN const std::vector<float*>&detRes, IN const unsigned int& clsNum, IN const unsigned int& infeHeight, IN const unsigned int& infeWidth, IN int* ptr)
{
    mSrcImage = srcImage;
    mDstImage = cv::Mat();
    mDetRes = detRes;
    mClsNum = clsNum;
    mInfeHeight = infeHeight;
    mInfeWidth = infeWidth;

    if (nullptr != ptr)
    {
        mPtr = new int();
        *mPtr = *ptr;
    }
}



/*
* -------------------------------------------->拷贝构造<-------------------------------------------- *
* Parameters:
*     input：
*         imgCrop:待拷贝的对象
*     output：
*         None
* returnValue:
*     None
* -------------------------------------------->拷贝构造<-------------------------------------------- *
*/
adaptiveImageCropping::adaptiveImageCropping(IN const adaptiveImageCropping& imgCrop)
{
    mSrcImage = imgCrop.mSrcImage;
    mDstImage = imgCrop.mDstImage;
    mDetRes = imgCrop.mDetRes;
    mClsNum = imgCrop.mClsNum;
    mInfeHeight = imgCrop.mInfeHeight;
    mInfeWidth = imgCrop.mInfeWidth;

    if (nullptr != imgCrop.mPtr)
    {
        mPtr = new int();
        *mPtr = *(imgCrop.mPtr);
    }
    
}



/*
* -------------------------------------------->赋值构造<-------------------------------------------- *
* Parameters:
*     input：
*         imgCrop:待复制的对象
*     output：
*         None
* returnValue:
*     None
* -------------------------------------------->赋值构造<-------------------------------------------- *
*/
adaptiveImageCropping& adaptiveImageCropping::operator= (IN const adaptiveImageCropping& imgCrop)
{
    mSrcImage = imgCrop.mSrcImage;
    mDstImage = imgCrop.mDstImage;
    mDetRes = imgCrop.mDetRes;
    mClsNum = imgCrop.mClsNum;
    mInfeHeight = imgCrop.mInfeHeight;
    mInfeWidth = imgCrop.mInfeWidth;


    if (nullptr != imgCrop.mPtr)
    {
        mPtr = new int();
        *mPtr = *(imgCrop.mPtr);
    }
}



/*
* -------------------------------------------->析构函数<-------------------------------------------- *
* Parameters:
*     input：
*         None
*     output：
*         None
* returnValue:
*     None
* -------------------------------------------->析构函数<-------------------------------------------- *
*/
adaptiveImageCropping::~adaptiveImageCropping()
{
    if (nullptr != mPtr)
    {
        delete mPtr;
    }
}



/*
* --------------------------------->获取目标矩形的的起点、终点坐标<--------------------------------- *
* Parameters:
*     input：
*         targetLabelCategory：指定的类别索引
*         thresLength：框宽度阈值
*     output：
*         targetCoord：保存 类别=targetLabelCategory的框(宽度最大的一个框)坐标，如果没有符合条件的框。则保存4个异常值-999
* returnValue:
*     None
* --------------------------------->获取目标矩形的的起点、终点坐标<--------------------------------- *
*/
int adaptiveImageCropping::getTargetCoord(IN const unsigned int& targetLabelCategory, IN const unsigned int& thresLength, OUT std::vector<float>& targetCoord)
{
    //判空以及判断数据的正确性
    if (0 == mDetRes.size())
    {
        std::cout << "mDetRes为空，请检查..." << std::endl;
        return -1;
    }
    else
    {
        if (6 != (sizeof(mDetRes) / mDetRes.size()) / sizeof(mDetRes[0][0]))
        {
            std::cout << "mDetRes[0]长度不正确..." << std::endl;
            return -1;
        }
    }
    
    //变量定义及初始化
    float temporaryLength = 0;
    float targetLength = 0;
    for (int i = 0; i < 4; ++i)
    {
        targetCoord.push_back(-999);
    }
    //按宽大小过滤目标的坐标,输出宽度最高的框的坐标
    for (int i = 0; i < mDetRes.size(); ++i)
    {
        if (targetLabelCategory == mDetRes[i][5])
        {
            targetLength = mDetRes[i][2] - mDetRes[i][0];
            if (thresLength <= targetLength)
            {
                if (targetLength > temporaryLength)
                {
                    for (int j = 0; j < 4; ++j)
                    {
                        targetCoord[j]=mDetRes[i][j];
                    }
                    temporaryLength = targetLength;
                }
            }
        }
    }

    return 0;
}



/*
* -------------------------------->根据检测结果统计每个类别目标数量<-------------------------------- *
* Parameters:
*     input：
*         clsNum：检测模型种类
*     output：
*         targetsNum:统计好的各类别数
* returnValue:
*     None
* -------------------------------->根据检测结果统计每个类别目标数量<-------------------------------- *
*/
int adaptiveImageCropping::countTargetNum(OUT std::vector<int>& targetsNum)
{
    //判空
    if (0 == mDetRes.size())
    {
        std::cout << "mDetRes为空，请检查..." << std::endl;
        return -1;
    }
    else
    {
        if (6 != (sizeof(mDetRes) / mDetRes.size()) / sizeof(mDetRes[0][0]))
        {
            std::cout << "mDetRes[0]长度不正确..." << std::endl;
            return -1;
        }
    }

    //输出变量初始化
    targetsNum.clear();
    if (mClsNum > 0)
    {
        targetsNum.push_back(0);
    }

    for (int i = 0; i < mDetRes.size(); ++i)
    {
        for(int clsIdx = 0; clsIdx < mClsNum; ++clsIdx)
        {
            if (clsIdx == mDetRes[i][5])
            {
                targetsNum[clsIdx]++;
            }
        }
    }

    return 0;
}



/*
* ------------------------------------------->自适应切图<------------------------------------------- *
* Parameters:
*     input：
*         targetLabelCategory：指定标签类别
*         thresLength：宽度阈值，过滤掉宽度小于该值的框
*         offsetHeight：高方向要扩大的像素个数，根据实际情况自适应扩大
*         offsetWidth：宽方向要扩大的像素个数，根据实际情况自适应扩大
*         dstImgHeight:填充后的图像高尺寸
*         dstImgWidth:填充后的图像宽尺寸
*     output：
*         None
* returnValue:
*     None
* ------------------------------------------->自适应切图<------------------------------------------- *
*/
int adaptiveImageCropping::adaptiveCropImage(IN const unsigned int& targetLabelCategory, IN const unsigned int& thresLength, IN const unsigned int& offsetHeight, IN const unsigned int& offsetWidth, IN const unsigned int& dstImgHeight, IN const unsigned int& dstImgWidth)
{
    //判空、判断输入的合法性
    if (mSrcImage.empty())
    {
        std::cout << "mSrcImage为空,请输入正确的图片..." << std::endl;
        return -1;
    }

    if (0 == mDetRes.size())
    {
        std::cout << "mDetRes为空,请检查..." << std::endl;
        return -1;
    }
    else
    {
        if (6 != (sizeof(mDetRes) / mDetRes.size()) / sizeof(mDetRes[0][0]))
        {
            std::cout << "mDetRes[0]长度不正确..." << std::endl;
            return -1;
        }
    }

    std::vector<float> targetCoord;
    getTargetCoord(targetLabelCategory, thresLength, targetCoord);

    targetCoordUpSampling(targetCoord);
    dilationTargetCoord(offsetHeight, offsetWidth, targetCoord);

    //注意：使用Rect或者Range切图内存共享，处理后的内存可能不连续，所以要加clone()重新开一段内存；
    mDstImage = mSrcImage(cv::Rect(targetCoord[0], targetCoord[1], targetCoord[2] - targetCoord[0], targetCoord[3] - targetCoord[1])).clone();
    //mDstImage = mSrcImage(cv::Range(targetCoord[1], targetCoord[3]), cv::Range(targetCoord[0], targetCoord[2])).clone();
    
    //自适应填充图像
    cv::Mat imageFilled;
    adaptivaFillImage(mDstImage, dstImgHeight, dstImgWidth, imageFilled);
    mDstImage = imageFilled;

    return 0;
}



/*
* -------------------------------------------->直接切图<------------------------------------------- *
* 注意：向下/右移动中心则offsetWidth和offsetHeight为正,否则为负数；
* Parameters:
*     input：
*         offsetWidth：图像宽中心偏移量
*         offsetHeight：图像高中心偏移量
*         dstImgHeight：目标图高大小
*         dstImgWidth：目标图宽大小
*     output：
*         dstImage：输出的目标图
* returnValue:
*     None
* -------------------------------------------->直接切图<------------------------------------------- *
*/
int adaptiveImageCropping::cropImage(IN const int& offsetWidth, IN const int& offsetHeight, IN const unsigned int& dstImgHeight, IN const unsigned int& dstImgWidth, OUT cv::Mat& dstImage)
{
    //判空、判断输入的合法性
    if (mSrcImage.empty())
    {
        std::cout << "mSrcImage为空,请输入正确的图片..." << std::endl;
        return -1;
    }

    if (mSrcImage.cols < dstImgWidth || mSrcImage.rows < dstImgHeight)
    {
        std::cout << "输入的目标尺寸太大，请检查..." << std::endl;
        return -1;
    }

    int rowCenter = mSrcImage.rows / 2 + offsetHeight;
    int colCenter = mSrcImage.cols / 2 + offsetWidth;

    int x = colCenter - dstImgWidth / 2;
    int y = rowCenter - dstImgHeight / 2;
    if (x < 0)
    {
        x = 0;
    }
    else
    {
        if (x + dstImgWidth > mSrcImage.cols - 1)
        {
            x = mSrcImage.cols - 1 - dstImgWidth;
        }
    }

    if (y < 0)
    {
        y = 0;
    }
    else
    {
        if (y + dstImgHeight > mSrcImage.rows - 1)
        {
            y = mSrcImage.rows - 1 - dstImgHeight;
        }
    }

    dstImage = mSrcImage(cv::Rect(x, y, dstImgWidth, dstImgHeight));

    return 0;
}



/*
* ---------------------------->自适应填充图片，防止图片resize时严重变形<---------------------------- *
* Parameters:
*     input：
*         srcImage：待填充的图像
*         dstImgHeight：目标图高度值
*         dstImgWidth：目标图宽度值
*     output：
*         dstImage:根据比例填充后的图像
* returnValue:
*     None
* ---------------------------->自适应填充图片，防止图片resize时严重变形<---------------------------- *
*/
int adaptiveImageCropping::adaptivaFillImage(IN const cv::Mat& srcImage, IN const unsigned int& dstImgHeight, IN const unsigned int& dstImgWidth, OUT cv::Mat& imgFilled)
{
    //判空、入参的合法性
    if (srcImage.empty())
    {
        std::cout << "输入的图像为空，请检查..." << std::endl;
        return -1;
    }

    //中心填充方式 填充图片
    int srcImgHeight = srcImage.rows;
    int srcImgWidth = srcImage.cols;
    unsigned int deltaSrcImgHeight = 0;
    unsigned int deltaSrcImgWidth = 0;
    int top = 0;
    int bottom = 0;
    int left = 0;
    int right = 0;

    float srcRatio = srcImgWidth / srcImgHeight;
    float dstRatio = dstImgWidth / dstImgHeight;
    if (srcRatio > dstRatio)    
    {
        deltaSrcImgHeight = int(srcImgWidth * dstImgHeight / dstImgWidth - srcImgHeight);
        top = deltaSrcImgHeight / 2;
        bottom = deltaSrcImgHeight - top;
    }
    else                        
    {
        deltaSrcImgWidth = int(srcImgHeight * dstImgWidth / dstImgHeight - srcImgWidth);
        left = deltaSrcImgWidth / 2;
        right = deltaSrcImgWidth - left;
    }
    
    if (3 == srcImage.channels())
    {
        cv::copyMakeBorder(srcImage, imgFilled, top, bottom, left, right, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    }
    else if (1 == srcImage.channels())
    {
        cv::copyMakeBorder(srcImage, imgFilled, top, bottom, left, right, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(255));
    }
    else
    {
        std::cout << "图像通道不正确，请检查..." << std::endl;
        return -1;
    }

    return 0;
}



/*
* --------------->对目标坐标扩大几个像素，并判断坐标是否到达原图的边缘，生成新的坐标<--------------- *
* Parameters:
*     input：
*         targetCoord：目标坐标
*     output：
*         targetCoord:扩大几个像素后的目标坐标
* returnValue:
*     None
* --------------->对目标坐标扩大几个像素，并判断坐标是否到达原图的边缘，生成新的坐标<--------------- *
*/
int adaptiveImageCropping::dilationTargetCoord(IN const unsigned int& offsetHeight, IN const unsigned int& offsetWidth, IN_OUT std::vector<float>& targetCoord)
{
    //判空、判断输入的合法性
    if (mSrcImage.empty())
    {
        std::cout << "mSrcImage为空,请输入正确的图片..." << std::endl;
        return -1;
    }

    if (0 == targetCoord.size())
    {
        std::cout << "输入的目标坐标为空,请检查..." << std::endl;
        return -1;
    }

    if (targetCoord[0] < 0)
    {
        std::cout << "输入的目标坐标数据异常，请检查..." << std::endl;
        return -1;
    }

    //计算目标扩大几个像素后的目标
    if (targetCoord[0] - offsetWidth >= 0)
    {
        targetCoord[0] = targetCoord[0] - offsetWidth;
    }
    else
    {
        targetCoord[0] = 0;
    }

    if (targetCoord[1] - offsetHeight >= 0)
    {
        targetCoord[1] = targetCoord[1] - offsetHeight;
    }
    else
    {
        targetCoord[1] = 0;
    }

    if (targetCoord[2] + offsetWidth <= mSrcImage.cols-1)
    {
        targetCoord[2] = targetCoord[2] + offsetWidth;
    }
    else
    {
        targetCoord[2] = mSrcImage.cols - 1;
    }

    if (targetCoord[3] + offsetHeight <= mSrcImage.rows - 1)
    {
        targetCoord[3] = targetCoord[3] + offsetHeight;
    }
    else
    {
        targetCoord[3] = mSrcImage.rows - 1;
    }

    return 0;
}



/*
* ------------------------------>将目标坐标进行上采样，恢复到原图尺寸<------------------------------ *
* Parameters:
*     input：
*         clsNum：检测模型种类
*     output：
*         targetsNum:统计好的各类别数
* returnValue:
*     None
* ------------------------------>将目标坐标进行上采样，恢复到原图尺寸<------------------------------ *
*/
int adaptiveImageCropping::targetCoordUpSampling(IN_OUT std::vector<float>& targetCoord)
{
    //判空、判断输入的合法性
    if (mSrcImage.empty())
    {
        std::cout << "mSrcImage为空,请输入正确的图片..." << std::endl;
        return -1;
    }

    if (0 == targetCoord.size())
    {
        std::cout << "输入的目标坐标为空,请检查..." << std::endl;
        return -1;
    }

    if (targetCoord[0] < 0)
    {
        std::cout << "输入的目标坐标数据异常，请检查..." << std::endl;
        return -1;
    }

    float heightRatio = mSrcImage.rows / mInfeHeight;
    float widthRatio = mSrcImage.cols / mInfeWidth;
    targetCoord[0] = widthRatio * targetCoord[0];
    targetCoord[1] = heightRatio * targetCoord[1];
    targetCoord[2] = widthRatio * targetCoord[2];
    targetCoord[3] = heightRatio * targetCoord[3];

    return 0;
}
