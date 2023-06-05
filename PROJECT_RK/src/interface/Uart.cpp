// ==========================================================
// 实现功能：读取消息的类源文件。1、从驱动板收消息并发送给主线程(然后由主线程分发给相机线程); 2、从主线程获取检测的各种消息，并发送给驱动板;
// 文件名称：Uart.cpp
// 相关文件：无
// 作   者：Liangliang Bai (liangliang.bai@leapting.com)
// 版   权：<Copyright(C) 2022-Leapting Technology Co.,LTD All rights reserved.>
// 修改记录：
// 日   期             版本       修改人   走读人  修改记录
// 2022.09.28   1.0.0.1  白亮亮                   None
// ==========================================================

#include "Uart.hpp"

//如果一个类内成员变量是static的，且需要将之设定为常量(const)，那么这个变量声明与初始化均可写在头文件内;
//如果一个类内成员变量是static的，但不需要将其设定为常量(const)，那么这个变量声明于头文件内，初始化(定义/实现)写在对应的cpp源文件中;
int Uart::m_waitFlag = -999;



// 函数功能：打开串口端口号;
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     dev_file：串口名称，如"/dev/ttyS3"就代表串口3;
// outputParas:
//     None
// returnValue:成功则返回打开的串口文件句柄fd,失败则返回-1;
// ----------------------------------->parameters<----------------------------------
int Uart::open_port_v1(INPUT char *dev_file)
{
    int fd = open(dev_file, O_RDWR|O_NOCTTY|O_NDELAY);
    if (-1 == fd)  
    {  
        perror ("serialport error\n");  
    }  
    else  
    {  
        printf ("open ");  
        printf ("%s", ttyname (fd));  
        printf (" succesfully\n");  
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



// 函数功能：设置串口相关参数
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     fd：打开的串口文件句柄;
//     nSpeed：串口速度（波特率）;
//     nBits：数据位，取值为7或者8;
//     nEvent：效验类型，取值为N,E,O,S;
//     nStop：停止位，取值为1或者2;
// outputParas:
//     None
// returnValue:设置成功返回0,失败则返回-1;
// ----------------------------------->parameters<----------------------------------
int Uart::set_opt(INPUT const int& fd, INPUT const int& nSpeed, INPUT const int& nBits, INPUT const char& nEvent, INPUT const int& nStop)
{
    struct termios newtio,oldtio;
    if ( tcgetattr( fd,&oldtio) != 0) 
    {
        perror("SetupSerial 1");
        return -1;
    }
    bzero( &newtio, sizeof( newtio ) );    //extern void bzero(void *s, int n)：置字节字符串的前n个字节为零
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

    //设置输入输出波特率
    switch( nSpeed )
    {
        case 2400:
            //int cfsetospeed(struct termios *termptr, speed_t speed);    如果成功返回0,否则返回-1 
            //参数：struct termios *termptr：指向termios结构的指针
            //     speed_t speed：需要设置的输出波特率  
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
    //在打开串口后，串口其实已经可以开始读取数据了 ，这段时间用户如果没有读取，将保存在缓冲区里，如果用户不想要开始的一段数据，或者发现缓冲区数据有误，可以使用这个函数清空缓冲
    // int tcflush（int filedes，int quene）
    // quene数该当是下列三个常数之一:TCIFLUSH  刷清输入队列
    //                           TCOFLUSH  刷清输出队列
    //                           TCIOFLUSH 刷清输入、输出队列
    tcflush(fd,TCIFLUSH);
    //int tcsetattr(int fd, int optional_actions, const struct termios *termios_p);  设置终端参数;成功返回0,失败返回-1;
    //    optional_actions取值：TCSANOW：不等数据传输完毕就立即改变属性。
    //                        ：TCSADRAIN：等待所有数据传输结束才改变属性。
    //                        ：TCSAFLUSH：清空输入输出缓冲区才改变属性。
    if((tcsetattr(fd,TCSANOW,&newtio))!=0)
    {
        perror("com set error");
        return -1;
    }
    printf("set done!\n");
    return 0;
}



// 函数功能：有参构造。将必须初始化的成员放在构造函数中初始化，将其他参数的初始化放在init()中初始化;
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     None
// outputParas:
//     None
// returnValue:None
// ----------------------------------->parameters<----------------------------------
Uart::Uart(INPUT char* portName)
{
    m_portName = portName;
    memset(m_bufferSend, 0, 8);
    m_fd = -1;
    m_len = 0;
    m_mesageFromDriverCard = 255;
    memset(m_messageFromMainThread, 255, 4);
    pthread_mutex_init(&m_uartMutex,NULL);
    pthread_mutex_init(&m_uartMutex2,NULL);
}



// 函数功能：析构函数，需要手动释放的资源均需要手动在析构中释放;
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     portName：串口号(如"/dev/ttyS3"就代表串口3);
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
Uart::~Uart()
{
    if(m_fd >=0)
    {
        close (m_fd);                                    //打开的端口不用的时候需要使用close(m_fd)关闭
    }

    // 销毁一个互斥锁即释放它所占用的资源，且要求锁当前处于开放状态。在Linux中，互斥锁并不占用任何资源,
    // 因此LinuxThreads中的pthread_mutex_destroy()除了检查锁状态以外(锁定状态则返回EBUSY)没有其他动作
    pthread_mutex_destroy(&m_uartMutex);
    pthread_mutex_destroy(&m_uartMutex2);


}



// 函数功能：专门的初始化函数;
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     None
// outputParas:
//     None
// returnValue:成功返回0,否则返回错误码;
// ----------------------------------->parameters<----------------------------------
int Uart::init()
{
    m_fd = open_port_v1(m_portName);                     //  "/dev/ttyS3", 串口3;如果是usb0："/dev/ttyusb0"
    CHECK_EXPR(m_fd < 0,-1);
    
    struct sigaction saio;
    saio.sa_handler = signal_handler_IO;                   //当收到信号就执行这个动作
    sigemptyset (&saio.sa_mask);  
    saio.sa_flags = 0;  
    saio.sa_restorer = NULL;  
    sigaction (SIGIO, &saio, NULL);  
    fcntl (m_fd, F_SETOWN, getpid ());                         //allow the process to receive SIGIO 
    fcntl (m_fd, F_SETFL, FASYNC);                                //make the file descriptor asynchronous  


    int ret = -1;
    ret = set_opt(m_fd,9600,8,'N',1);
    CHECK_EXPR(ret < 0,-1);

    cout << "uart::init finished!" << endl; 

    //将this传入pthread_create函数(就相当于)start_thread函数内部创建的对象指向了this，这样在类外通过替他对象调用时，相当于调用的this->方法;                             
    int res = pthread_create(&m_uartTid,NULL,start_thread,(void*)this);

    return 0;
}



// 函数功能：开启消息线程的入口函数;
// 注意：静态成员里面不允许使用this指针(this指针只能用于非静态成员函数)，所以才又创建了一个对象，并将其指向this，这样在外面其他对象(假设对象A)调用该函数时，
//      实际上是用this操作其他数据，就跟直接使用A对象操作数据一样的。
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     param：用于接收this，该参数由创建线程的函数pthread_create作为第三个参数传入;
// outputParas:
//     None
// returnValue:None
// ----------------------------------->parameters<----------------------------------
void *Uart::start_thread(INPUT void *param)
{
    Uart *uart = (Uart *)param;
    uart->run();
}



// 函数功能：将消息发送到主线程(主线程会主动调用该函数，使用共享内存的方式发数据);
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     None
// outputParas:
//     signalValue：获取到的信号值;
// returnValue:None
// ----------------------------------->parameters<----------------------------------
void Uart::sendMsgToMainThread(OUTPUT unsigned char& signalValue)
{
    pthread_mutex_lock(&m_uartMutex);
    if(255 != m_mesageFromDriverCard)
    {
        signalValue = m_mesageFromDriverCard;
        std::cout<<"消息线程发消息到主线程"<<int(m_mesageFromDriverCard)<<std::endl;
        m_mesageFromDriverCard = 255;
    }
    pthread_mutex_unlock(&m_uartMutex);
}



// 函数功能：将主线程的数据传递到消息线程(主线程会主动调用该函数，使用共享内存的方式);
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     None
// outputParas:
//     signalValue：获取到的信号值;
// returnValue:None
// ----------------------------------->parameters<----------------------------------
void Uart::getMsgFromMainThread(INPUT unsigned char* signalValues)
{
    //注意：数组作为函数参数传递的就是个指针地址，无法使用sizeof方法求数组的长度。如果用其他指针接收了数组的首地址，也无法使用该方法求长度。
    //基于上原因，数组长度只能显示地作为函数参数传入函数中。所以此处干脆就不检查了，所以传入的时候要非常小心，防止导致越界错误。

    pthread_mutex_lock(&m_uartMutex2);
    m_messageFromMainThread[0] = signalValues[0];
    m_messageFromMainThread[1] = signalValues[1];
    m_messageFromMainThread[2] = signalValues[2];
    m_messageFromMainThread[3] = signalValues[3];
    pthread_mutex_unlock(&m_uartMutex2);
}



// 函数功能：收发消息。从驱动板收消息存在共享内存中，将主线程发到共享内存的信息发给驱动板;
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     None
// outputParas:
//     None
// returnValue:返回错误/异常值
// ----------------------------------->parameters<----------------------------------
int Uart::run()
{
    unsigned char buf[8];                    //接收驱动板发来的整包数据的临时字符段
    int i = 0;

    while (true)  
    {
        //收消息:从驱动板收消息并放在消息缓存中，等待另一个线程去取 
        if (0 == m_waitFlag)  
        {
            memset (buf, 0, sizeof(buf));
            int messageLen = receive_from_stm32(buf, 8);      
            if(0 != messageLen)
            {
                pthread_mutex_lock(&m_uartMutex);
                m_mesageFromDriverCard = buf[5];
                pthread_mutex_unlock(&m_uartMutex);
            }

            m_waitFlag = -999;               //读完发来的消息就置为-999,等待下次变为0,改变信号，程序就不再执行，等待下次变为0，
        } 

        //发消息，从另一个线程获取到图像算法结果，根据结果值发消息给驱动板
        if(255!=m_messageFromMainThread[0] || 255!=m_messageFromMainThread[1] || 255!=m_messageFromMainThread[2] || 255!=m_messageFromMainThread[3])
        {
            pthread_mutex_lock(&m_uartMutex2);
            
            send_to_stm32(m_messageFromMainThread, 4);
            i++;
            memset(m_messageFromMainThread, 255, 4);
            pthread_mutex_unlock(&m_uartMutex2);
        }
        usleep(200000);                   //消息线程运行太快了，为了防止与驱动板互相收发消息出问题，需要将消息线程堵塞超过200ms
    }  
    return 0;
    
}



// 函数功能：校验，将前6字节送入该校验函数，校验当前这包数据是否正常;
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     buffer：待收/发的消息;
//     size：消息字节个数，定义为6;
// outputParas:
//     None
// returnValue:返回2个字节的校验变换值，这个2个字节再变换后与buffer的后两个字节做对比;
// ----------------------------------->parameters<----------------------------------
unsigned short Uart::ByteCrc16(INPUT unsigned char *buffer, INPUT unsigned short size)
{
    unsigned char i = 0;
    unsigned short j = 0, crc = 0xFFFF;

    for (j = 0; j < size; j++) 
    {
        crc ^= buffer[j];
        for (i = 0; i < 8; i++) 
        {
            crc = (crc >> 1) ^ ((crc & 1) ? 0xA001 : 0);
        }
    }
    return ((crc & 0xFF) << 8 | (crc & 0xFF00) >> 8);
}



// 函数功能：校验，将前6字节送入该校验函数，校验当前这包数据是否正常;
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     pdata：待收/发的消息(该项目定义为8个字节);
//     len：消息字节个数，值为8;
// outputParas:
//     None
// returnValue:返回发送的字节个数;
// ----------------------------------->parameters<----------------------------------
int Uart::send_base(INPUT unsigned char *pdata,INPUT const int& len)
{
    int nwrite = write(m_fd,pdata,len);
    CHECK_EXPR(nwrite != len,-1);

    return nwrite;
}



// 函数功能：发送1字节真正的消息值(该函数会将这1个字节消息按协议组包成一包8字节的数据，然后发送给驱动板);
// 注意：m_bufferSend的8字节按照协议每一位放对应的内容;
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     pdata：真正要发送的1字节消息;
//     len：消息字节个数，值为1;
// outputParas:
//     None
// returnValue:成功返回0,否则返回-1;
// ----------------------------------->parameters<----------------------------------
int Uart::send_to_stm32(INPUT unsigned char *pdata, INPUT const int& len)
{   
    m_bufferSend[m_len++] = 0xff;              // 固定值
    m_bufferSend[m_len++] = 0x10;              // 固定值
    m_bufferSend[m_len++] = 0x08;              // 存放地址
    m_bufferSend[m_len++] = 0x04;              // 存放地址
    m_bufferSend[m_len++] = 0x00;    
    m_bufferSend[m_len++] = 0x02;
    m_bufferSend[m_len++] = 0x04; 

    for (int i = 0;i < len;i++) 
    {
        m_bufferSend[m_len++] = pdata[i];      // 存放要发送的值
    }
    
    unsigned short crc16 = ByteCrc16(m_bufferSend,m_len);
    m_bufferSend[m_len++] = (crc16 >> 8);      //由校验返回值获取
    m_bufferSend[m_len++] = (crc16 & 0x00ff);  //由校验返回值获取

    int nwrite = send_base(m_bufferSend,m_len);
    if (nwrite != m_len) 
    {
        printf("send_to_stm32 failed!\n");
        return -1;
    }
    m_len = 0;

    return 0;
}



// 函数功能：从串口获取驱动板发来的8字节数据,将获取到的数据进行变换并校验其正确性;
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     pdata：获取的8字节消息;
//     len：获取的消息的字节个数，值为8;
// outputParas:
//     None
// returnValue:成功返回消息长度(8),否则返回-1;
// ----------------------------------->parameters<----------------------------------
int Uart::receive_from_stm32(OUTPUT unsigned char *pdata,INPUT const int& len)
{
    int nread = receive_base(pdata,len);
    if (nread != len) 
    {
        printf("receive_from_stm32 failed!\n");
        return -1;
    }
    unsigned short crc16 = ByteCrc16(pdata,len-2);
    if((pdata[len-1] == (crc16&0xff)) && (pdata[len-2] == (crc16>>8)))
    {
        return len;
    }
    return -1;
}



// 函数功能：从串口获取驱动板发来的8字节数据;
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     pdata：获取的8字节消息;
//     len：获取的消息的字节个数，值为8;
// outputParas:
//     None
// returnValue:返回读取的数据的字节个数;
// ----------------------------------->parameters<----------------------------------
int Uart::receive_base(unsigned char *pdata,int len)
{
    int nread = read(m_fd,pdata,len);
    CHECK_EXPR(nread != len,-1);

    return nread;
}



// 函数功能：static函数.如该驱动板要发消息过来，会有一个通知信号。将该函数(以函数指针的方式，由于信号参数的类型原因，导致该函数只能是类的static函数)与信号绑定，
//         一收到信号该函数就会执行。m_waitFlag一置为0,其他函数就会去串口读消息;
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     status：该参数不需要我们手动传入;
// outputParas:
//     None
// returnValue:None
// ----------------------------------->parameters<----------------------------------
void Uart::signal_handler_IO (INPUT int status)  
{  
    printf ("--------------signal_handler_IO:将要收到驱动版发来的消息---------------.\n");  
    m_waitFlag = 0;  
}  

