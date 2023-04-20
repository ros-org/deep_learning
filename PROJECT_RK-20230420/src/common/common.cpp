#include "common.h"


static struct timeval startTime,endTime;  
static float Timeuse;


void print_int_vector(const char *title,vector<int > v)
{   
    int line_cnt = 6;
    cout << "Print " << title << ":" << endl;
    for (int i = 0;i < v.size();i++) {
        if (i % line_cnt == 0) {
            cout << endl;
        }
        cout <<  v[i] << " ";
    }
    cout << endl;
}

void print_vector(const char *title,vector<Dtype > v)
{   
    int line_cnt = 6;
    cout << "Print " << title << ":" << endl;
    for (int i = 0;i < v.size();i++) {
        if (i % line_cnt == 0) {
            cout << endl;
        }
        cout <<  v[i] << " ";
    }
    cout << endl;
}

void Timer::start()
{
    gettimeofday(&m_startTime,NULL);  
}

float Timer::end(char *title)
{
    gettimeofday(&m_endTime,NULL);  
    m_Timeuse = 1000.*(m_endTime.tv_sec - m_startTime.tv_sec) + (m_endTime.tv_usec - m_startTime.tv_usec)/1000.;  
    printf("[%s] Timeuse = %f ms\n",title,m_Timeuse);
    return m_Timeuse;
}


//记录起始时间
void cal_time_start()
{
    gettimeofday(&startTime,NULL);  
}

//记录结束时间，输出时间差，与cal_time_start函数结合使用；
float cal_time_end(const char *title)
{
    gettimeofday(&endTime,NULL);  
    Timeuse = 1000.*(endTime.tv_sec - startTime.tv_sec) + (endTime.tv_usec - startTime.tv_usec)/1000.;  
    printf("[%s] Timeuse = %f ms\n",title,Timeuse);
    
    return Timeuse;
}

int writeTxtFile(const char *filename,Dtype *pData,int size)
{    
    //char fullFilePath[128];
    
    //switch_char((char *)filename,'/','-');
    //snprintf(fullFilePath,sizeof(fullFilePath),"out/%s",filename);
	FILE *fp = fopen(filename,"w+");
    //cout << "filename:" << filename << endl;
	CHECK_EXPR(fp == NULL,-1);
    
	char data_str[64];
	int cnt = 0;
	for (int i = 0;i < size;i++) {
		snprintf(data_str,sizeof(data_str),"%-12f",(float)pData[i]);
		int nwrite = fwrite(data_str,sizeof(char),strlen(data_str),fp);
        CHECK_EXPR(nwrite != strlen(data_str),-1);

		if (cnt % 64 == 0 && cnt != 0) {
			int nwrite = fwrite("\n",sizeof(char),1,fp);
            CHECK_EXPR(nwrite != 1,-1);
		}
		cnt++;
	}
	fclose(fp);
}

Dtype *load_data(const char *filename, int size)
{
	printf("begin load file:%s\n", filename);

	FILE *fp = fopen(filename, "rb");
	CHECK_EXPR(fp == NULL, NULL);

	fseek(fp, 0L, SEEK_END);
	unsigned long length = ftell(fp);

	printf("size * sizeof(Dtype):%ld\n", size * sizeof(Dtype));
	printf("length:%ld\n", length);

	CHECK_EXPR(size * sizeof(Dtype) != length, NULL);
	fseek(fp, 0L, SEEK_SET);

	Dtype *pData = (Dtype *)malloc(sizeof(Dtype) * size);
	CHECK_EXPR(pData == NULL, NULL);

	int nread = fread(pData, sizeof(Dtype), size, fp);
	CHECK_EXPR(nread != size, NULL);

	printf("load file:%s and size:%d\n", filename, size);

	return pData;
}


bool check_file_exists(char *filepath)
{
    if((access(filepath,F_OK))!=-1)   {   
        return true; 
    } else {
        return false; 
    }
}