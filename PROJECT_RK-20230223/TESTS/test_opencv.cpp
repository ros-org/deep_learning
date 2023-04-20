#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;

int test_rtsp(void);
int save_video();


int main(int argc,char **argv)
{
	test_rtsp();
    
    return 0;
}

int test_rtsp(void)
{
	Mat  img;
	cout << "RK Project.........." << endl;
	String source = "rtsp://admin:Litian123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0";//"test.mp4";
        VideoCapture captrue;//(source);
	captrue.open(source);
	if (!captrue.isOpened()) {
		printf("---------------------captrue opencv failed!\n");
		return 1;
	} else {
		printf("Open captrue success!\n");
	}

	VideoWriter write;
	// std::string outFlie = "1_out.avi";
	int frames = static_cast<int>(captrue.get(CAP_PROP_POS_FRAMES));
	cout << "frames:" << frames << endl;

	int w = static_cast<int>(captrue.get(CAP_PROP_FRAME_WIDTH));
	int h = static_cast<int>(captrue.get(CAP_PROP_FRAME_HEIGHT));
	cout << "w:" << w << endl;
	cout << "h:" << h << endl;
	cv::Size S(w, h);
	double r = captrue.get(CAP_PROP_FPS);
	write.open("/userdata/output/res.mp4", VideoWriter::fourcc('M', 'P', 'E', 'G'), 10, S, true);
    
	cv::Mat frame;
	int cnt = 0;
	char filename[64];
    while(1) {
        captrue >> frame;
        if (frame.empty())
            break;
		// snprintf(filename,sizeof(filename),"output/%d.jpg",cnt);
		// cv::imwrite(filename,frame);
        // cv::imshow("hello", frame);
        // cv::waitKey(50);
		write.write(frame);
		printf("%d...\n",cnt);
		cnt++;
    }
	write.release();
	return 0;
}
