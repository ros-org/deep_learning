#include <stdio.h>
#include "Configer.hpp"
#include "Yolo.hpp"
#include "common.h"
#include "ProcessMgr.hpp"
#include <iostream>
#include <fstream>
#include "Configer.hpp"

using namespace std;

void test()          // 为了测试开机自启动是否生效，
{
    ofstream ofs;
    ofs.open("hhh007.txt", ios::out);
    for (int i = 1; i < 264347; i++)
    {
        ofs << "Q"
            << "\t" << i << "\t" << endl;
    }
    ofs.close();
}

int test_img(int argc, char **argv);


void weatherModelOfflineTest()
{
    //模型初始化
    Configer *pConfiger = Configer::GetInstance();
    std::cout << "version:" << pConfiger->verison() << std::endl;

    std::cout<<"Loading weather classification model..."<<std::endl;
    CLA_CFG_t* m_p_cla_weather_cfg;
    m_p_cla_weather_cfg = pConfiger->get_cla_weather_cfg();
    Classify mCla_weather;
    int ret = mCla_weather.init(m_p_cla_weather_cfg);

    //图像预处理 
    int col_center = 640;
    int row_center = 360 + 130;
    cv::Mat frame, im_classify_weather_part, im_classify_weather;
    std::string imgDir = "/userdata/data/";
    std::vector<String> imgNamesPath;
    glob(imgDir, imgNamesPath, false);  //调用opncv中的glob函数，将遍历路径path，将该路径下的全部文件名的绝对路径存进imgNmaes

    for(int imgNumIdx=0; imgNumIdx<imgNamesPath.size(); ++imgNumIdx)
    {
        std::cout<<"当前图片路径:"<<imgNamesPath[imgNumIdx]<<std::endl;
        frame = imread(imgNamesPath[imgNumIdx],cv::IMREAD_UNCHANGED);
        im_classify_weather_part = frame(cv::Rect(col_center-240, row_center-112, 480, 224));
        cv::resize(im_classify_weather_part, im_classify_weather, cv::Size(m_p_cla_weather_cfg ->feed_w,m_p_cla_weather_cfg ->feed_h), (0, 0), (0, 0), cv::INTER_LINEAR);
        
        int classifyRes;
        uint8_t chwImg[3*m_p_cla_weather_cfg ->feed_w*m_p_cla_weather_cfg ->feed_h];
        HWC2CHW(im_classify_weather, chwImg);
        int ret = mCla_weather.run(chwImg, "CHW", classifyRes);
        std::cout<<"推理结果="<<classifyRes<<std::endl;
    }
}



int test_img(int argc, char **argv)
{
    Configer *pConfiger = Configer::GetInstance();

    std::cout << "version:" << pConfiger->verison() << std::endl;
    Yolo mYolo;
    int ret = -1;

    YOLO_CFG_t *p_yolo_cfg = pConfiger->get_yolo_cfg();
    p_yolo_cfg->model = "/userdata/models/detection.rknn";

    mYolo.init(p_yolo_cfg);

    // const char *model_path = argv[1];
    const char *img_path = "/userdata/data/test.jpg";

    // Load image
    cv::Mat orig_img = cv::imread(img_path, 1);

    cv::Mat img = orig_img.clone();
    if (!orig_img.data)
    {
        printf("cv::imread %s fail!\n", img_path);
        return -1;
    }
    cal_time_start();
    if (orig_img.cols != p_yolo_cfg->feed_w || orig_img.rows != p_yolo_cfg->feed_h)
    {
        // printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, p_yolo_cfg->feed_w, p_yolo_cfg->feed_h);
        cv::resize(orig_img, img, cv::Size(p_yolo_cfg->feed_w, p_yolo_cfg->feed_h), (0, 0), (0, 0), cv::INTER_LINEAR);
    }
    cal_time_end("resize");

    cal_time_start();
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cal_time_end("cvtColor");

    vector<float *> res;
    ret = mYolo.run(img.data, "CHW", res);
    CHECK_EXPR(ret != 0, -1);

    mYolo.show_res(img, res);

    return 0;
}



//天气模型离线批量测试
int main888()
{
    weatherModelOfflineTest();
    std::cout<<"测试结束."<<std::endl;
}



//清洗机视觉框架运行
int main(int argc, char **argv)
{
#if 1
    // 由于C语言pthread_create函数的原因，导致在C语言中启动线程的方式看着很难受（不直接优雅美观）
    ProcessMgr *ProcessMgr = ProcessMgr::GetInstance();
    ProcessMgr->start();

    // 阻塞主线程
    while (1)
    {
        pause();
    }
#else
    test_img(argc, argv);
#endif
    return 0;
}