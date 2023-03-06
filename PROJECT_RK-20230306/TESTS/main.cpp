#include <stdio.h>
#include "Configer.hpp"
#include "Yolo.hpp"
#include "common.h"
#include "ProcessMgr.hpp"
#include <iostream>
#include <fstream>
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

int main(int argc, char **argv)
{
    cout<< "AAAAAAAAAAAAAAAAAAA" << endl;
    test();
#if 1
    ProcessMgr *ProcessMgr = ProcessMgr::GetInstance();
    ProcessMgr->start();

    while (1)
    {
        pause();
    }
#else
    test_img(argc, argv);
#endif
    return 0;
}

int test_img(int argc, char **argv)
{
    Configer *pConfiger = Configer::GetInstance();

    std::cout << "version:" << pConfiger->verison() << std::endl;
    Yolo mYolo;
    int ret = -1;

    YOLO_CFG_t *p_yolo_cfg = pConfiger->get_yolo_cfg();
    p_yolo_cfg->model = "/userdata/models/lt_yolov5s.rknn";

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
    ret = mYolo.run(img.data, res);
    CHECK_EXPR(ret != 0, -1);

    mYolo.show_res(img, res);

    return 0;
}
