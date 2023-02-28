#include <stdio.h>
#include <stdlib.h>
#include <iostream>


#ifdef __cplusplus 
extern "C"
{
#endif
/*Include ffmpeg header file*/
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
 
#include "libavutil/imgutils.h" 
#include "libavutil/opt.h"     
#include "libavutil/mathematics.h"  
#include "libavutil/samplefmt.h"
#ifdef __cplusplus
}
#endif

using namespace std;
 
int main(int argc,char **argv)
{
    AVFormatContext *pFormatCtx;
    char filepath[] = "rtsp://admin:Litian123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0";
    AVPacket *packet;

    //初始化
    av_register_all();
    avformat_network_init();
    pFormatCtx = avformat_alloc_context();
    AVDictionary* options = NULL;
    av_dict_set(&options, "buffer_size", "1024000", 0); //设置缓存大小,1080p可将值跳到最大
    av_dict_set(&options, "rtsp_transport", "tcp", 0); //以udp的方式打开,
    av_dict_set(&options, "stimeout", "5000000", 0); //设置超时断开链接时间，单位us
    av_dict_set(&options, "max_delay", "500000", 0); //设置最大时延
    packet = (AVPacket *)av_malloc(sizeof(AVPacket));
 
    //打开网络流或文件流
    if (avformat_open_input(&pFormatCtx, filepath, NULL, NULL) != 0) {
        printf("Couldn't open input stream.\n");
        return -1;
    }
    
    //查找码流信息
    if (avformat_find_stream_info(pFormatCtx, NULL)<0) {
        printf("Couldn't find stream information.\n");
        return -1;
    }
    
    //查找码流中是否有视频流
    int videoindex = -1;
    unsigned i = 0;
    for (i = 0; i<pFormatCtx->nb_streams; i++) {
        if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoindex = i;
            break;
        }
    }
        
    if (videoindex == -1)
    {
        printf("Didn't find a video stream.\n");
        return -1;
    }
 
    //保存一段的时间视频，写到文件中
    FILE *fpSave;
    fpSave = fopen("/userdata/output/rtsp_receive.h265", "wb");
    for (i = 0; i < 100; i++)   //这边可以调整i的大小来改变文件中的视频时间
    {
        if (av_read_frame(pFormatCtx, packet) >= 0)
        {
            if (packet->stream_index == videoindex)
            {
                fwrite(packet->data, 1, packet->size, fpSave);  
            }
            av_packet_unref(packet);
        }
        cout << "save " << i << endl;
    }
 
    fclose(fpSave);
    av_free(pFormatCtx);
    //av_free(packet);
    av_packet_unref(packet);

    return 0;
}
