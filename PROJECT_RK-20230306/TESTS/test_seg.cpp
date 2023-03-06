/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <stdio.h>
#include "Configer.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "Segmentation.hpp"

using namespace std;
using namespace cv;

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    int ret = -1;

    cout << "test_seg .........." << endl;
    Configer *pConfiger = Configer::GetInstance();
    SEG_CFG_t *p_seg_cfg = pConfiger->get_seg_cfg();
    
    Segmentation seg;
    ret = seg.init(p_seg_cfg);
    CHECK_EXPR(ret != 0,-1);

    const char *img_path = "/userdata/data/test.jpg";
    Mat im = cv::imread(img_path, 1);
    ASSERT(im.empty() == false);

    Mat im1;
    cv::cvtColor(im, im1, cv::COLOR_BGR2RGB);
    // im1 = im;

    Mat im_feed;
    cv::resize(im1, im_feed, cv::Size(p_seg_cfg->feed_w, p_seg_cfg->feed_h),(0, 0), (0, 0), cv::INTER_LINEAR);

    Mat seg_res;

    // unsigned char *p_seg;
    ret = seg.run(im_feed.data,seg_res);
    CHECK_EXPR(ret != 0,-1);

    Mat im_draw = Mat::zeros(im_feed.rows, im_feed.cols, CV_8UC3);
    seg.draw_seg(im_draw,seg_res);

    cout << "version 11111111111111111111111" << endl;
    cv::imwrite("/userdata/data/unetpp_test_result.jpg",im_draw);

    return 0;
}