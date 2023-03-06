#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <errno.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <stdlib.h>
#include "Uart.hpp"

int set_opt(int fd,int nSpeed, int nBits, char nEvent, int nStop)
{
    struct termios newtio,oldtio;
    if ( tcgetattr( fd,&oldtio) != 0) {
        perror("SetupSerial 1");
        return -1;
    }
    bzero( &newtio, sizeof( newtio ) );
    newtio.c_cflag |= CLOCAL | CREAD;
    newtio.c_cflag &= ~CSIZE;
    switch( nBits )
    {
        case 7:
            newtio.c_cflag |= CS7;
            break;
        case 8:
            newtio.c_cflag |= CS8;
            break;
    }
    switch( nEvent )
    {
        case 'O':
            newtio.c_cflag |= PARENB;
            newtio.c_cflag |= PARODD;
            newtio.c_iflag |= (INPCK | ISTRIP);
            break;
        case 'E':
            newtio.c_iflag |= (INPCK | ISTRIP);
            newtio.c_cflag |= PARENB;
            newtio.c_cflag &= ~PARODD;
            break;
        case 'N':
            newtio.c_cflag &= ~PARENB;
            break;
    }
    switch( nSpeed )
    {
        case 2400:
            cfsetispeed(&newtio, B2400);
            cfsetospeed(&newtio, B2400);
            break;
        case 4800:
            cfsetispeed(&newtio, B4800);
            cfsetospeed(&newtio, B4800);
            break;
        case 9600:
            cfsetispeed(&newtio, B9600);
            cfsetospeed(&newtio, B9600);
            break;
        case 115200:
            cfsetispeed(&newtio, B115200);
            cfsetospeed(&newtio, B115200);
            break;
        default:
            cfsetispeed(&newtio, B9600);
            cfsetospeed(&newtio, B9600);
            break;
    }
    if( nStop == 1 )
        newtio.c_cflag &= ~CSTOPB;
    else if ( nStop == 2 )
    newtio.c_cflag |= CSTOPB;
    newtio.c_cc[VTIME] = 0;
    newtio.c_cc[VMIN] = 0;
    tcflush(fd,TCIFLUSH);
    if((tcsetattr(fd,TCSANOW,&newtio))!=0)
    {
        perror("com set error");
        return -1;
    }
    printf("set done!\n");
    return 0;
}

int open_port_v1(char *dev_file)
{
    int fd = open(dev_file, O_RDWR|O_NOCTTY|O_NDELAY);
    if (-1 == fd){
        perror("Can't Open Serial Port");
        return(-1);
    }
    else
        printf("open %s .....\n",dev_file);

    if(fcntl(fd, F_SETFL, 0))
        printf("fcntl failed!\n");
    else
        printf("fcntl=%d\n",fcntl(fd, F_SETFL,0));
    if(isatty(STDIN_FILENO)==0)
        printf("standard input is not a terminal device\n");
    else
        printf("isatty success!\n");
    printf("fd-open=%d\n",fd);
    return fd;
}


int open_port(int fd,int comport)
{
    char *dev[]={"/dev/ttyS1","/dev/ttyS2","/dev/ttyS3"};
    long vdisable;
    if (comport==1)
    {    fd = open( "/dev/ttyS1", O_RDWR|O_NOCTTY|O_NDELAY);
        if (-1 == fd){
            perror("Can't Open Serial Port");
            return(-1);
        }
        else
            printf("open ttyS0 .....\n");
    }
    else if(comport==2)
    {    fd = open( "/dev/ttyS2", O_RDWR|O_NOCTTY|O_NDELAY);
        if (-1 == fd){
            perror("Can't Open Serial Port");
            return(-1);
        }
        else
            printf("open ttyS2 .....\n");
    }
    else if (comport==3)
    {
        fd = open( "/dev/ttyS1", O_RDWR|O_NOCTTY|O_NDELAY);
        if (-1 == fd){
            perror("Can't Open Serial Port");
            return(-1);
        }
        else
            printf("open ttyS1 .....\n");
    }
    if(fcntl(fd, F_SETFL, 0))
        printf("fcntl failed!\n");
    else
        printf("fcntl=%d\n",fcntl(fd, F_SETFL,0));
    if(isatty(STDIN_FILENO)==0)
        printf("standard input is not a terminal device\n");
    else
        printf("isatty success!\n");
    printf("fd-open=%d\n",fd);
    return fd;
}

#if 1
int main(int argc,char **argv)
{
    Uart uart;
    uart.init();
    char buffer[1024];
    static int cnt = 0;

    unsigned short test_data = 0x234f;
    printf("data1:0x%x\n",(test_data  >> 8));
    printf("data2:0x%x\n",(test_data  & 0x00ff));
    
    while (1) {
        snprintf(buffer,sizeof(buffer),"send %d\n",cnt++);
        // uart.send_base((unsigned char *)buffer,strlen((const char *)buffer));
        printf("send %s\n",buffer);
        usleep(1000000);
    }
    return 0;
}
#else
int main(int argc,char **argv)
{
    char *dev_file = "/dev/ttyS3";
    int fd = open_port_v1(dev_file);
    if (fd < 0){
        perror("Can't Open Serial Port\n");
        return(-1);
    }
    else {
        printf("open %s ..... success\n",dev_file);
    }
    int i = -1;
    if((i=set_opt(fd,115200,8,'N',1)) ){
        perror("set_opt error");
        return 0;
    }
    printf("fd=%d\n",fd);
    
    while (1) {

        char *test_str = "test123456";
        int nwrite = write(fd,test_str,strlen(test_str));
        printf("write=%d,%s\n",nwrite,test_str);
        usleep(1000000);
        printf("send %s",test_str);
    }
    return 0;
}
#endif