#include"getCurrentTime.h"


void tm_YMDHMS::changeTmToYmdhms(tm& t)     //将tm格式数据转tm_YMDHMS格式。获取当前的年-月-日 时：分：秒时间
	{
		tm_year = t.tm_year + 1900;
		tm_mon = t.tm_mon;
		tm_mday = t.tm_mday;
		tm_hour = t.tm_hour;
		tm_min = t.tm_min;
		tm_sec = t.tm_sec;
	}

	void tm_YMDHMS::changeYmdhmsToTm(tm& t) const  //将tm_YMDHMS格式数据转tm格式
	{
		t.tm_year = tm_year - 1900;
		t.tm_mon = tm_mon - 1;
		t.tm_mday = tm_mday;
		t.tm_hour = tm_hour;
		t.tm_min = tm_min;
		t.tm_sec = tm_sec;
	}
