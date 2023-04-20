#include<string>
#include<iostream>

//将struct tm转tm_YMDHMS
struct tm_YMDHMS
{
	int tm_sec;                                                                      /* seconds after the minute - [0,59] */
	int tm_min;                                                                     /* minutes after the hour - [0,59] */
	int tm_hour;                                                                   /* hours since midnight - [0,23] */
	int tm_mday;                                                                 /* day of the month - [1,31] */
	int tm_mon;                                                                   /* months since January - [1,12] */
	int tm_year;                                                                   /* years since 0000 */

	//Default constructor withput parameters:Initialize using the initializer list
	tm_YMDHMS() : tm_sec(0), tm_min(0), tm_hour(0), tm_mday(0), tm_mon(0), tm_year(0) {};
	tm_YMDHMS(int sec, int min, int hour, int day, int month, int year) : tm_sec(sec), tm_min(min), tm_hour(hour), tm_mday(day), tm_mon(month), tm_year(year) {};

	void changeTmToYmdhms(tm& t);                     //将tm格式数据转tm_YMDHMS格式。获取当前的年-月-日 时：分：秒时间

	void changeYmdhmsToTm(tm& t) const ;       //将tm_YMDHMS格式数据转tm格式

};