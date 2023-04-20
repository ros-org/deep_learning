#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <sys/time.h>  
#include <string.h>
#include <stdlib.h>
#include <unistd.h>   
#include <fcntl.h>   

typedef float  Dtype;
using namespace std;

#define VOS_MAX(a,b) (a)>(b)?(a):(b)
#define VOS_MIN(a,b) (a)>(b)?(b):(a)
#define DIM_OF(arr) sizeof(arr)/sizeof(arr[0])

#define ASSERT(expr) if (!(expr)) { \
                printf("[ASSERT] %s:%d error!",__FUNCTION__,__LINE__); \
                exit(-1); \
            } 
            
#define CHECK_EXPR(expr,ret) if (expr) { \
                printf("[CHECK] %s:%d error!",__FUNCTION__,__LINE__); \
                return ret; \
            } 

#define CHECK_EXPR_NO_RETURN(expr) if (expr) { \
                printf("[CHECK] %s:%d error!",__FUNCTION__,__LINE__); \
                getchar(); \
                return; \
            } 
            

class Timer
{
public:
    Timer(){};
    ~Timer(){};
    void start();
    float end(char *title);

private:
    struct timeval m_startTime;
    struct timeval m_endTime;  
    float m_Timeuse;
};

void cal_time_start();
float cal_time_end(const char *title);
Dtype *load_data(const char *filename, int size);
int writeTxtFile(const char *filename,Dtype *pData,int size);
void print_vector(const char *title,vector<Dtype > v);
void print_int_vector(const char *title,vector<int > v);
bool check_file_exists(char *filepath);


#endif