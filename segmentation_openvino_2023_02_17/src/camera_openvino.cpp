
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h> 
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <dynamic_reconfigure/server.h>
// #include "camera_tensorrt/camera_tensorrtConfig.h"
// 上面的 camera_tensorrtConfig.h，似乎已经用 CMakeLists.txt 中的 
// cfg/camera_tensorrt.cfg 代替，是否还需要？先注释掉。

#include "openvinoInferenceUnetpp.h"

class camera_openvino
{
public:
	camera_openvino();

private:
  void init_param();
	void depth_callback(const sensor_msgs::PointCloud2ConstPtr &img_msg);
	void img_callback(const sensor_msgs::ImageConstPtr &img_msg);
	void trig_callback(const std_msgs::HeaderConstPtr &msg);

	ros::NodeHandle nh_;
	ros::NodeHandle ph_;

	ros::Subscriber sub_depth;
	ros::Subscriber sub_img;
	ros::Subscriber sub_trig;

	std::string ns_;
	int trig_;
	std::string offline_path_;
	int skip_;
	int skip_count_;

	ros::Publisher pub_image;
	ros::Publisher pub_depth;
	//Deep learning model global params
	void * buffer = nullptr;
	void * context = nullptr;
	float* hostInputBuffer=nullptr;
	ltTensorParams unetppTensorParams;

	int counte = 0;
	int testnum = 0;
	sensor_msgs::PointCloud2ConstPtr depth_msg;

	void paramCallback(camera_tensorrt_cfg::camera_tensorrtConfig& config, uint32_t level);

    dynamic_reconfigure::Server<camera_tensorrt_cfg::camera_tensorrtConfig> server;
    dynamic_reconfigure::Server<camera_tensorrt_cfg::camera_tensorrtConfig>::CallbackType f;
};

camera_openvino::camera_openvino()
:ph_("~")
{
	init_param();
	// 在构造函数中初始化参数，然后加载推理模型。
	loadModel(unetppTensorParams, &buffer, &context, &hostInputBuffer);

	sub_depth = nh_.subscribe("camera/depth/depth_raw", 1000, &camera_openvino::depth_callback, this);
	sub_img = nh_.subscribe("camera/rgb/rgb_raw", 1000, &camera_openvino::img_callback, this);
	sub_trig = nh_.subscribe("trig", 1000, &camera_openvino::trig_callback, this);
	//ros::Subscriber sub_state=nh_.subscribe("/trig",100,state_callback);
	pub_image = nh_.advertise<sensor_msgs::Image>("camera/rgb/rgb_" + ns_, 1);
	pub_depth = nh_.advertise<sensor_msgs::PointCloud2>("camera/depth/depth_" + ns_, 1000);
	// 循环等待回调函数
	ROS_INFO(" Ready to show tl_robot_vision informtion.");

    f = boost::bind(&camera_openvino::paramCallback, this, _1, _2);
    server.setCallback(f);
}

void camera_openvino::init_param()
{
	trig_ = 0;
	ph_.param("ns", ns_, std::string("solar_panel"));
	int imgC, imgH, imgW, classNum;
	std::string modelFilePath;
    ph_.param("imgC", imgC, 3);
    ph_.param("imgH", imgH, 400);
    ph_.param("imgW", imgW, 640);
    ph_.param("modelFilePath", modelFilePath, std::string("~/xulei_ws/camera_tensorrt/cfg/400x640.trt"));
    if ('~' == modelFilePath[0])
    {
        modelFilePath = getenv("HOME") + modelFilePath.substr(1);
    }
    ph_.param("classNum", classNum, 2);
	unetppTensorParams.inputTensorNames.push_back("inputs");
	unetppTensorParams.batchSize = 1;
	unetppTensorParams.outputTensorNames.push_back("outputs");
	unetppTensorParams.imgC = imgC;
	unetppTensorParams.imgH = imgH;
	unetppTensorParams.imgW = imgW;
	unetppTensorParams.modelFilePath = modelFilePath;
	unetppTensorParams.classNum = classNum;

    ph_.param("offline_path", offline_path_, std::string(""));
    ph_.param("skip", skip_, 15);
    skip_count_ = 0;
}

void camera_openvino::depth_callback(const sensor_msgs::PointCloud2ConstPtr &img_msg){
	
	//pub img_msg
	depth_msg = img_msg;
	// pub_depth.publish(depth_msg);

}

void camera_openvino::img_callback(const sensor_msgs::ImageConstPtr &img_msg){
	
	if (trig_)
	{	
		if (skip_count_ != 0)
		{
			skip_count_++;
			if (skip_count_ >= skip_)
			{
				skip_count_ = 0;
			}
			return;
		}

		testnum ++;
		std::cout<< "run times:"<< testnum <<std::endl;

		sensor_msgs::PointCloud2 sync_depth_msg = *depth_msg;

		cv_bridge::CvImageConstPtr ptr;
		ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::RGB8);
		cv::Mat show_img;

		// --------------------------------图像算法调用--------------------------------//
		//------------》吸清洗机的模型推理《------------//
		cv::Mat singleSegOut1;

		if (offline_path_ != "")
		{
			show_img = cv::imread(offline_path_);
		}
		else
		{
			show_img = ptr->image;
		}
		// 把 show_img 送入模型中，使用 OpenVino 推理，返回分割后的结果 singleSegOut1 。
		inference_online_openvino(show_img,  unetppTensorParams, buffer, context, hostInputBuffer, singleSegOut1);
		// trtInferenceOnLine(show_img,  unetppTensorParams, buffer, context, hostInputBuffer, singleSegOut1);
		sensor_msgs::ImagePtr msg; 
		msg = cv_bridge::CvImage(img_msg->header, "mono8", singleSegOut1).toImageMsg();
		pub_image.publish(msg);
		pub_depth.publish(depth_msg);
		//------------》吸清洗机的模型推理《------------//
		if (trig_ == 1)
		{
			trig_ = 0;
		}
		else
		{
			skip_count_++;
		}
	}
}

void camera_openvino::trig_callback(const std_msgs::HeaderConstPtr &msg)
{
	if (ns_ == msg->frame_id)
	{
		trig_ = msg->seq;
	}
}

void camera_openvino::paramCallback(camera_tensorrt_cfg::camera_tensorrtConfig& config, uint32_t level)
{
    trig_ = config.trig;
	if (config.trig == 1)
	{
		config.trig = 0;
	}
	offline_path_ = config.offline_path;
}

int main(int argc,char** argv)
{
    ros::init(argc, argv, "robot_visison_server");
    camera_openvino camera_openvino;
		// ros::spin 会让订阅者不断检查接收到的消息包，根据收到的消息包不同，运行不同的 callback 程序。
    ros::spin();
}
