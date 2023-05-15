#ifndef __CONFIGER_HPP__
#define __CONFIGER_HPP__

#include <iostream>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>

typedef struct 
{
    char *model;
    int feed_w;
    int feed_h;
    int cls_num;
}YOLO_CFG_t;


typedef struct 
{
    char *model;
    int feed_w;
    int feed_h;
    int cls_num;
}SEG_CFG_t;


typedef struct        //分类结构体 CLA_CFG_t 被typedef 修饰之后就可以定义CLA_CFG_t 类型的变量了，
{
    char *model;      // 分类模型
    int feed_w;       // 分类模型输入尺寸的宽
    int feed_h;       // 分类模型输入尺寸的高
    int cls_num;      // 分类模型的类别数目
}CLA_CFG_t;           // 定义了一个名字为 CLA_CFG_t 的结构体，


class Configer        // 定义配置程序类
{
public:
    static Configer *GetInstance();
    Configer(){};                   // 构造函数，
    ~Configer(){};                  // 析构函数，
    int init();
    char *verison() 
    {
        return m_version;        // m 代表类成员变量
    }

    YOLO_CFG_t *get_yolo_cfg() 
    {
        return &m_yolo_cfg;      // 返回检测配置的参数
    };

    YOLO_CFG_t *get_yolo_cfg2() 
    {
        return &m_yolo_cfg2;      // 返回检测配置的参数
    };

    SEG_CFG_t *get_seg_cfg() 
    {
        return &m_seg_cfg;
    };
    
    CLA_CFG_t *get_cla_weather_cfg()
    {
        return &m_cla_weather_cfg;       // 返回天气分类模型的配置参数
    }


    // 清洁度参数
    CLA_CFG_t * get_cla_cfg()
    {
        return &m_cla_cfg;       // 返回分类模型的配置参数
    }
    
    static Configer *mpConfiger;
    
private:
    char *m_version;
    YOLO_CFG_t m_yolo_cfg, m_yolo_cfg2;
    SEG_CFG_t m_seg_cfg;
    CLA_CFG_t m_cla_weather_cfg;
    CLA_CFG_t m_cla_cfg;      //结构体CLA_CFG_t 的实例化对象是 配置类 Configer  的 一个类属性  
    
    bool m_init;
};

#endif