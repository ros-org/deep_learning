#include "Configer.hpp"

Configer *Configer::mpConfiger = NULL;

Configer *Configer::GetInstance()
{
    if (mpConfiger == NULL) 
    {
        mpConfiger = new Configer();
        mpConfiger->init();
    }
    return mpConfiger;
}


int Configer::init()
{
    m_version = "LT-RK V0.11";

    /**************YOLO parameters**************/
    m_yolo_cfg.feed_h = 288;                                       //输入图像的height
    m_yolo_cfg.feed_w = 512;                                       //输入图像的width
    m_yolo_cfg.cls_num = 3;                                        //类别总数
    m_yolo_cfg.model = "/userdata/models/yolov5s.rknn";            //yolov5s rknn模型路经
    m_init = true;
    std::cout << "finish Configer::init ..." << std::endl;
    std::cout << "Configer ..." << m_version << std::endl;
    /**************YOLO parameters**************/


    /**************UNET parameters**************/
    m_seg_cfg.feed_h = 416;                                        //输入图像的height
    m_seg_cfg.feed_w = 512;                                        //输入图像的width
    m_seg_cfg.cls_num = 2;                                         //类别数，包括背景
    m_seg_cfg.model = "/userdata/models/unetpp.rknn";              //unetpp rknn模型路径
    /**************UNET parameters**************/

    /*************RESNET weather parameters*************/
    m_cla_weather_cfg.feed_h = 112;
    m_cla_weather_cfg.feed_w = 112;
    m_cla_weather_cfg.cls_num = 5;
    m_cla_weather_cfg.model = "/userdata/models/resnet.rknn";      // 天气分类模型文件路径
    /*************RESNET weather parameters*************/


    
    /*************RESNET parameters*************/
    m_cla_cfg.feed_h = 112;
    m_cla_cfg.feed_w = 112;
    m_cla_cfg.cls_num = 2;
    m_cla_cfg.model = "/userdata/models/resnet-bll.rknn";
    /*************RESNET parameters*************/
    return 0;
}