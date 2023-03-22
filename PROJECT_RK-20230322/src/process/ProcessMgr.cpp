#include "ProcessMgr.hpp"
#include "segResPostProcessing.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <numeric>
#include <chrono>     //用于测试时间，该方式更精准

using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

//外面的对象调用GetInstance函数，可以将该全局对象返赋给外面的对象
ProcessMgr *ProcessMgr::mpProcessMgr = NULL;
ProcessMgr *ProcessMgr::GetInstance()
{
    if (mpProcessMgr == NULL) 
    {
        mpProcessMgr = new ProcessMgr();                         //new 创建对象，并调用构造函数，
        mpProcessMgr->init();
    }
    return mpProcessMgr;                                         // 返回主线程类对象地址，
}

int ProcessMgr::init()
{
    int ret = -1;                                                // 初始化一个变量，
    char* portName = "/dev/ttyS3";                               // RKNN 烧写系统后 dev下的端口号，没有 ttyS3 说明系统有问题，
    m_uart = Uart(portName);                                     // 通信端口号，
    m_bdebug = true;                                             // 调试模式标志
    m_b_detect = true;                                           // 检测模型标志
    m_b_seg = true;                                              // 分割算法标志
    m_b_cla = true;                                              // 清洁度分类标志
    m_b_weatherClassification = true;                            // 是否运行天气分类

    mpConfiger = Configer::GetInstance();                         

    if (m_b_detect) 
    {
        m_p_yolo_cfg = mpConfiger->get_yolo_cfg();
        ret = mYolo.init(m_p_yolo_cfg);
        CHECK_EXPR(ret != 0,-1);
    }

    if (m_b_seg) 
    {
        m_p_seg_cfg = mpConfiger->get_seg_cfg();
        ret = mSeg.init(m_p_seg_cfg);
        CHECK_EXPR(ret != 0,-1);
    }

    if (m_b_weatherClassification) 
    {
        m_p_cla_weather_cfg = mpConfiger->get_cla_weather_cfg();
        ret = mCla_weather.init(m_p_cla_weather_cfg);
        CHECK_EXPR(ret != 0,-1);
    }


    if (m_b_cla) 
    {
        m_p_cla_cfg = mpConfiger->get_cla_cfg();
        ret = mCla.init(m_p_cla_cfg);
        CHECK_EXPR(ret != 0,-1);
    }

    ret = m_CapProcess.init();    // 在该函数中开启了获取图像的线程
    CHECK_EXPR(ret != 0,-1);

    m_uart.init();
    if (m_bdebug == true) 
    {
        m_debug_fd = open("/userdata/output/debug.txt",O_RDWR | O_TRUNC | O_CREAT);
        CHECK_EXPR(m_debug_fd < 0,-1);

        char *data = "Debug ..............\n";
        int size = strlen(data);
        int nwrite = write(m_debug_fd,data,strlen(data));
        CHECK_EXPR(nwrite != size,-1);
    }
    return 0;
}

void ProcessMgr::writeMsgToLogfile(const std::string& strMsg,  unsigned char info)
{
    tm_YMDHMS currentTime;
	struct tm * localTime;
	time_t nowtime;
	time(&nowtime);                                                                   //得到当前系统时间
	localTime = localtime(&nowtime);                                //将nowtime变量中的日历时间转化为本地时间，存入到指针为p的时间结构体中
    currentTime.changeTmToYmdhms(*localTime);    //Change tm format data into tm_YMDHMS;
	
     char* logBuff = new char[1024];
    snprintf(logBuff,1024,"%d-%d-%d-%d-%d-%d: %s:%d;\n", currentTime.tm_year,currentTime.tm_mon,currentTime.tm_mday,currentTime.tm_hour, currentTime.tm_min,currentTime.tm_sec,strMsg.c_str(), int(info));
    // std::cout<<logBuff<<std::endl;
    int nwrite = write(m_debug_fd,logBuff,strlen(logBuff));
    if(nullptr!=logBuff)
    {
        delete [] logBuff;
        logBuff = nullptr;
    }
}

void ProcessMgr::writeMsgToLogfile2(const std::string& strMsg,  float info)
{
    tm_YMDHMS currentTime;
	struct tm * localTime;
	time_t nowtime;
	time(&nowtime);                                                                   //得到当前系统时间
	localTime = localtime(&nowtime);                                //将nowtime变量中的日历时间转化为本地时间，存入到指针为p的时间结构体中
    currentTime.changeTmToYmdhms(*localTime);    //Change tm format data into tm_YMDHMS;
	
     char* logBuff = new char[1024];
    snprintf(logBuff,1024,"%d-%d-%d-%d-%d-%d: %s:%d;\n", currentTime.tm_year,currentTime.tm_mon,currentTime.tm_mday,currentTime.tm_hour, currentTime.tm_min,currentTime.tm_sec,strMsg.c_str(), info);
    // std::cout<<logBuff<<std::endl;
    int nwrite = write(m_debug_fd,logBuff,strlen(logBuff));
    if(nullptr!=logBuff)
    {
        delete [] logBuff;
        logBuff = nullptr;
    }
}

void *ProcessMgr::start_thread(void *param)
{
    ProcessMgr *pProcessMgr = (ProcessMgr *)param;
    pProcessMgr->run();
    return NULL;
}

int ProcessMgr::start()
{
    int res = pthread_create(&m_tid,NULL,start_thread,(void*)this);
    return 0;
}

int ProcessMgr::save_image(Mat &im,char *title)
{
    char filename[64];
    static int cnt = 0;
    snprintf(filename,sizeof(filename),"output/%s_%d.jpg",title,cnt);
    printf("==========%d...\n",cnt);
    cnt++;
    return 0;
} 

int ProcessMgr::run()
{
    Mat frame;                                                  //拷贝缓冲区的图片，用于检测和分割
    int col_center;                                             //frame的中心col坐标
    int row_center;                                             //frame的中心row坐标
    Mat im_classify_weather;                                    //用于天气分类的图
    Mat im_classify_cleanliness;                                //用于清洁度分类的图
    Mat im_classify_cleanliness_part;                           //将清洁度分类的图切一块进行推理
    Mat im_detect;                                              //用于检测的图
    Mat im_seg;                                                 //用于分割的图
    int det_cnt = 0;                                            //检测模型运行次数
    int seg_cnt = 0;                                            //分割模型运行次数
    int cla_cnt = 0;                                            //分类运行次数
    int inerval_cnt = 1;                                        //预测帧率(从缓冲区拿到的图片中每隔inerval_cnt检测一次)间隔
    int cnt = 0;                                                //从缓冲区真正拿到图的数量
    float angleThresValue = 10.;                                //角度阈值，当角度不小于该值，认为无法通过
    Timer timer;
    int classifyRes = -999;                                     //定义天气分类的类别结果
    int cleanlinessOutput = -999;                               //清洁度单次推理结果。 最脏的类别为：1；越干净类别数越大；
    std::vector<float> cleanlinessOutputs;                      //清洁度多次推理结果，当收到相机掉头信号时将结果清零(即清洁度计算单趟清洁的结果)
    float cleannessQuaRes;                                      //清洁度量化结果
    Mat seg_res;                                                //分割后的结果图（索引图）
    char buffer[1024];                                          //用于存储被格式化后的路径等字符串
    static int debug_seg_cnt = 0;                               //分割模型运行次数，用于日志中             
    static int debug_det_cnt = 0;                               //检测模型运行次数，用于日志中                                               
    unsigned char WeatherInfo = 0x01;                           //发送给驱动板是否可以正常出去工作的消息（沙尘暴等极端天气不可出去工作），1代表不可以出去工作
    unsigned char WeatherInfo_2 = 0x05;                         //发送给驱动板是否可以正常出去工作的消息，5和254代表可以出去工作
    unsigned char detInfo = 0x01;                               //发送给驱动板的检测(有断裂)消息
    unsigned char segInfo = 0x02;                               //发送给驱动板的分割(角度太大)消息
    unsigned char speedDown = 0x03;                             //发送给驱动板的 要下坡了，请减速
    unsigned char restoreSpeed = 0x04;                          //发送给驱动板的 下坡结束，请恢复到原来的速度
    bool flag_isRunSegModel = false;                            //是否运行分割模型的标志，检测到桥架框并且无断裂才进行分割;
    bool lastDownhillStatus = false;                            //上次检测到的 是否下桥状态，和最新图的下桥架状态做对比，结果不一致，则发消息并更新状态 
    bool lastfractureStatus = false;                            //上次是否断裂的状态
    bool lastAngleStatus = false;                               //上次角度是否太大的状态
    unsigned char signalFromMsgThread = 255;                    //从消息线程获取到的消息
    unsigned char msgsTomsgThread[4] = {254, 100, 254, 254};    //存储检测的各种信息的数组，当其中有值发生变化就立刻发送到消息线程，254是视觉询问驱动板专用数值(当所以值为254时，认为一切正常)；
    high_resolution_clock::time_point beginTime;                //起始时间
    high_resolution_clock::time_point endTime;                  //结束时间
    milliseconds timeInterval;                                  //时间间隔

    while (1) 
    {
        //注意：模块3的顺序不能写在模块1和2之前；
        //----------------------->1、从缓存区获取图像<------------------------//
        bool have_data = m_CapProcess.run(frame);
        if (have_data == false) 
        {
            usleep(5000);
            continue;
        }
        //----------------------->1、从缓存区获取图像<------------------------//


        //--------------> 2、间隔inerval_cnt次取到的图 进行检测<---------------//
        cnt++;
        if (cnt % inerval_cnt != 0) 
        {
            continue;
        }
        //--------------> 2、间隔inerval_cnt次取到的图 进行检测<---------------//


        //--------------------------->3、消息交互<---------------------------//
        //3、1、主动先发消息给消息线程
        m_uart.getMsgFromMainThread(msgsTomsgThread);
        std::cout<<"msgsTomsgThread[0]="<<(int)msgsTomsgThread[0]<<std::endl;
        std::cout<<"msgsTomsgThread[1]="<<(int)msgsTomsgThread[1]<<std::endl;
        std::cout<<"msgsTomsgThread[2]="<<(int)msgsTomsgThread[2]<<std::endl;
        std::cout<<"msgsTomsgThread[3]="<<(int)msgsTomsgThread[3]<<std::endl;
        memset(msgsTomsgThread, 254, 4);

        beginTime = high_resolution_clock::now();
        ProcessMgr::writeMsgToLogfile("Main thread send msg to message thread[Ask if there is a turn-around message reply]", WeatherInfo_2);
        
        //3.2、从消息线程获取驱动板发来的消息
        m_uart.sendMsgToMainThread(signalFromMsgThread);
        //3.3、将从消息线程拿到的转头消息发给相机线程
        if(255!=signalFromMsgThread)
        {
            //当相机掉头说明单趟从新开始，则清洁度重新计算；
            cleanlinessOutputs.clear();

            //1.4、调用相机的对外消息接口，并将该消息写入到相机线程，然后在相机线程中控制云台运动
            m_CapProcess.getMsgFromMainThread(signalFromMsgThread);
            ProcessMgr::writeMsgToLogfile("Get message from Driver Card", signalFromMsgThread);
            signalFromMsgThread = 255;
        }
        //--------------------------->3、消息交互<---------------------------//


        //--------------------------->4、天气分类<---------------------------//
        // 注意：将所有天气分为多个类别，可以工作的类别排布在前面（如1-3），不可工作的类别排布在后面（如4-5）
        if(m_b_weatherClassification)
        {  
            std::cout<<"-----------------------------------------Running weather classification model-----------------------------------------"<<std::endl;  
            
            cv::resize(frame, im_classify_weather, cv::Size(m_p_cla_weather_cfg ->feed_h,m_p_cla_weather_cfg ->feed_h), (0, 0), (0, 0), cv::INTER_LINEAR);
            cv::cvtColor(im_classify_weather, im_classify_weather, cv::COLOR_BGR2RGB);
            int ret = mCla_weather.run(im_classify_weather.data, classifyRes);

            std::cout<<"天气分类结果: "<< classifyRes<<std::endl; 
            if(classifyRes >= 3) 
            {
                msgsTomsgThread[0] = WeatherInfo;
                ProcessMgr::writeMsgToLogfile("Send msg to message thread [the weather is not good for going out to work]", WeatherInfo);
            }
            else
            {
                msgsTomsgThread[0] = WeatherInfo_2;
                ProcessMgr::writeMsgToLogfile("Send msg to message thread [the weather is good for going out to work]", WeatherInfo_2);
            }                                  
            m_b_weatherClassification = false;    //天气每次上电只需要检测一次
        }
        //--------------------------->4、天气分类<---------------------------//


        //------------------------>5、分类：清洁度检测<-----------------------//
        if(m_b_cla)
        {
            std::cout<<"-----------------------------------------Running classification model-----------------------------------------"<<std::endl;
            timer.start();
            //5.1图像预处理
            col_center = frame.cols/2;
            row_center = frame.rows/2+100;
            std::cout<<"col_center="<<col_center<<" row_center="<<row_center<<std::endl;
            im_classify_cleanliness_part = frame(cv::Rect(col_center-112, row_center-112, 224, 224));
            cv::resize(im_classify_cleanliness_part, im_classify_cleanliness, cv::Size(m_p_cla_cfg->feed_w, m_p_cla_cfg->feed_h), (0, 0), (0, 0), cv::INTER_LINEAR);
            // cv::cvtColor(im_classify_cleanliness, im_classify_cleanliness, cv::COLOR_BGR2RGB);
            
            //5.2清洁度分类推理
            int ret = mCla.run(im_classify_cleanliness.data, cleanlinessOutput);
            std::cout<<"当前清洁度推理结果:"<<cleanlinessOutput<<std::endl;
            CHECK_EXPR(ret != 0,-1);
            timer.end("Cla");

            //5.3、统计单趟清洁度，清洗机运行一趟时清洁度是实时变化的，当第二趟开始，重新进行计算；
            cleanlinessOutputs.push_back(cleanlinessOutput*(100.0/m_p_cla_cfg->cls_num));
            cleannessQuaRes = accumulate(cleanlinessOutputs.begin(), cleanlinessOutputs.end(), 0.0)/cleanlinessOutputs.size();
            uchar cleannessQuaResPercent = int(cleannessQuaRes);

            //5.4、实时发送清洁度到消息线程
            msgsTomsgThread[1] = cleannessQuaResPercent;
            ProcessMgr::writeMsgToLogfile("Send msg to message thread [What is the cleanness value]", cleannessQuaResPercent);
        }
        //------------------------>5、分类：清洁度检测<-----------------------//


        //---------------------------->6、检测<-----------------------------//
        if (true == m_b_detect) 
        {
            flag_isRunSegModel = false;
            ProcessMgr::writeMsgToLogfile2("------------------------->Running object detect model<--------------------------:", det_cnt);
            det_cnt++;

            timer.start();
            cv::resize(frame, im_detect, cv::Size(m_p_yolo_cfg->feed_w, m_p_yolo_cfg->feed_h), (0, 0), (0, 0), cv::INTER_LINEAR);
            cv::cvtColor(im_detect, im_detect, cv::COLOR_BGR2RGB);

            vector<float *> res;
            res.clear();
            int ret = mYolo.run(im_detect.data,res);
            CHECK_EXPR(ret != 0,-1);
            timer.end("Detect");
            
            // 根据最终的检测结果，统计每个类别的数量
            int bridgeNum = 0;                                     // 当前图片检测出的bridge数量
            int fractureNum = 0;                                  // 当前图片检测出的fracture数量
            int lowerBridgeNum = 0;                         // 当前图片检测出的lowerBridge数，一般应该为0或者1
            for (int i = 0;i < res.size();i++) 
            {
                if (res[i][5] == 0.)                                      // 0代表bridge，[bridge,fracture,lowerBridge]
                {
                    bridgeNum++;
                }

                if (res[i][5] == 1.)                                       // 1代表fracture
                {
                    fractureNum++;
                }

                if (res[i][5] == 2.)                                       // 2代表lowerBridge，将要下桥架
                {
                    lowerBridgeNum++;
                }
            }

            // 是否将要下桥的状态判断。两次状态不一样则发消息并更新状态;
            bool latestDownhillStatus = false;        //当前最新图 是否将要下桥架的状态，true为将要下桥架;
            if(lowerBridgeNum > 0)
            {
                latestDownhillStatus = true;
            }

            if(lastDownhillStatus != latestDownhillStatus)
            {
                if(lastDownhillStatus==false && latestDownhillStatus==true)
                {
                    msgsTomsgThread[2] = speedDown;
                    ProcessMgr::writeMsgToLogfile("发送减速的消息给消息消息：请减速:", speedDown);
                }
                else
                {
                    msgsTomsgThread[2] = restoreSpeed;
                    ProcessMgr::writeMsgToLogfile("发送恢复速度的消息给消息线程:", restoreSpeed);
                }
                // 更新状态
                lastDownhillStatus = latestDownhillStatus;
            }

            //是否有断裂 状态判断，两次状态不一致则发消息。这样可以做到 只发一次断裂/无断裂的消息给驱动板
            bool latestFractureStatus = false;              //当前最新图 是否桥架断裂的状态，true为断裂;
            if(fractureNum > 0)
            {
                latestFractureStatus = true;
            }

            if(lastfractureStatus != latestFractureStatus)
            {
                if(lastfractureStatus==false && latestFractureStatus==true)
                {
                    msgsTomsgThread[3] = detInfo;
                    ProcessMgr::writeMsgToLogfile("发送检测结果到消息线程：上次无断裂，这次有断裂", detInfo);
                }
                else
                {
                    ProcessMgr::writeMsgToLogfile2("发送检测结果到消息线程：上次断裂，这次无断裂", 666);
                }

                lastfractureStatus = latestFractureStatus;
            }

            //没检测到断裂且检测到有桥架，分割标识符置为true
            if(0 == fractureNum && bridgeNum>=1)
            {
                flag_isRunSegModel = true;
                ProcessMgr::writeMsgToLogfile("没检测到断裂且检测到有桥架,分割标识符置为true", flag_isRunSegModel);
            }
 
            //保存日志及检测结果图
            if (m_bdebug == true && debug_det_cnt < 1024)
            {
                for (int i = 0;i < res.size();i++) 
                {  
                    snprintf(buffer,sizeof(buffer),"[detect %d] %f %f %f %f %f %f\n",debug_det_cnt,
                             res[i][0],res[i][1],res[i][2],res[i][3],
                             res[i][4],res[i][5]);
                    int size = strlen(buffer);
                    int nwrite = write(m_debug_fd,buffer,strlen(buffer));
                    CHECK_EXPR(nwrite != size,-1);
                }
                
                if (res.size() >= 0)               
                {
                    mYolo.show_res(im_detect,res);                                                                                                                           //画框并打印检测结果
                    snprintf(buffer,sizeof(buffer),"/userdata/output/lt_det_out_%d.jpg",debug_det_cnt++);
                    cv::imwrite(buffer,im_detect);                                                                                                                               //保存检测图片
                }
            }
        }
        //---------------------------->6、检测<-----------------------------//
 

        //---------------------------->7、分割<-----------------------------//
        if (true==m_b_seg && true==flag_isRunSegModel) 
        {
            ProcessMgr::writeMsgToLogfile2("---------------------->Running seg model<----------------------", seg_cnt);
            seg_cnt++;

            timer.start();
            cv::resize(frame, im_seg, cv::Size(m_p_seg_cfg->feed_w, m_p_seg_cfg->feed_h), 
                                                        (0, 0), (0, 0), cv::INTER_LINEAR);
            cv::cvtColor(im_seg, im_seg, cv::COLOR_BGR2RGB);
            int ret = mSeg.run(im_seg.data,seg_res);
            CHECK_EXPR(ret != 0,-1);
            timer.end("Seg");

            //处理分割结果，计算桥架角度
            float angle = -999.;
            #if 1
            ret = (int)postProcessingSegRes(seg_res,angle);
            ProcessMgr::writeMsgToLogfile2("分割后计算角度结果", angle);

            bool latestAngleStatus = false;
            if(angle >= angleThresValue)
            {
                latestAngleStatus = true;
            }

            if(lastAngleStatus != latestAngleStatus)
            {
                if(lastAngleStatus==false && latestAngleStatus==true)
                {
                    msgsTomsgThread[3] = segInfo;
                    ProcessMgr::writeMsgToLogfile("发送分割角度结果的消息(上次可以通过，本次无法通过)到消息线程", segInfo);
                }
                else
                {
                    ProcessMgr::writeMsgToLogfile2("发送分割角度结果的消息(上次无法通过，本次可以通过)到消息线程", 666);
                }

                lastAngleStatus = latestAngleStatus;
            }
            #endif

            //写日志及将分割结果图保存到硬盘上
            if (m_bdebug == true && debug_seg_cnt < 1024) 
            {
                snprintf(buffer,sizeof(buffer),"[seg %d]angle:%f\n",debug_seg_cnt,angle);
                cout << buffer;
                int size = strlen(buffer);
                int nwrite = write(m_debug_fd,buffer,strlen(buffer));
                CHECK_EXPR(nwrite != size,-1);

                Mat im_draw =  Mat::zeros(seg_res.rows, seg_res.cols, CV_8UC3);
                mSeg.draw_seg(im_draw,seg_res);
                snprintf(buffer,sizeof(buffer),"/userdata/output/lt_seg_out_%d.jpg",debug_seg_cnt++);
                cv::imwrite(buffer,im_draw);
                
                // // 下面这种方法需要的时间更长，但是可以将分割结果图画在原图上
                // timer.start();
                // std::vector<std::vector<cv::Point>> seg_res_contour;
                // std::vector<cv::Vec4i> hreir;
                // cv::findContours(seg_res, seg_res_contour, hreir, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE,cv::Point());
                // drawContours(im_seg, seg_res_contour, -1, cv::Scalar(255, 0, 0),1, 8);
                // snprintf(buffer,sizeof(buffer),"/userdata/output/segMask_%d.jpg",debug_seg_cnt++);
                // cv::imwrite(buffer,im_seg);
                // timer.end("BBBBBBBBBBBBBB");
            }
        }
        //---------------------------->7、分割<-----------------------------//


        //-------------------------->8、阻塞主线程<--------------------------//
        //注意：当主线程比消息线程快的时候，就无法保证上次的消息真正发出去，所以此处要稍微阻塞一下主线程
        //8、1.计算上次检测总共的时间
        endTime = high_resolution_clock::now();      
        timeInterval = std::chrono::duration_cast<milliseconds>(endTime - beginTime);
        std::cout<<"Running time: "<<timeInterval.count()<<"ms"<<std::endl;
        //8.2、阻塞主线程（当主线程运行一次的时间小于200ms，则阻塞）
        if(timeInterval.count()<300000)
        {
            usleep(300000-timeInterval.count());  //注意：usleep单位是微妙
        }
        //-------------------------->8、阻塞主线程<--------------------------//            
    }

    return 0;
}

