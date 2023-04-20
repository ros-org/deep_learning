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


typedef struct 
{
    char *model;
    int feed_w;
    int feed_h;
    int cls_num;
}CLA_CFG_t;


class Configer
{
public:
    static Configer *GetInstance();
    Configer(){};
    ~Configer(){};
    int init();
    char *verison() 
    {
        return m_version;
    }

    YOLO_CFG_t *get_yolo_cfg() 
    {
        return &m_yolo_cfg;
    };

    SEG_CFG_t *get_seg_cfg() 
    {
        return &m_seg_cfg;
    };

    CLA_CFG_t* get_cla_cfg()
    {
        return &m_cla_cfg;
    }
    
    static Configer *mpConfiger;
    
private:
    char *m_version;
    YOLO_CFG_t m_yolo_cfg;
    SEG_CFG_t m_seg_cfg;
    CLA_CFG_t m_cla_cfg;
    bool m_init;
};

#endif