#include "detection_output.h"
#include <math.h>
#include <map> 

void DetectionOutput::init(int in_width,int in_height,int num_cls)
{
	m_in_width = in_width;
	m_in_height = in_height;

	m_num_cls = num_cls;
	m_max_num1 = 1024;
	m_conf_thres = 0.7;
	m_iou_thres = 0.5;
	
	gen_anchors();
	
	m_innersize = m_num_cls + 5;
	m_predFilter1 = (float *)malloc(sizeof(float) * m_max_num1 * 6);

	m_cls_boxes.clear();
	m_cls_boxes_finnal.clear();
	for (int i = 0;i < m_num_cls;i++) 
	{
		vector<float *> data;
		data.clear();
		m_cls_boxes.push_back(data);

		vector<float *> data1;
		data1.clear();
		m_cls_boxes_finnal.push_back(data1);
	}

	/////////////////nms 
	m_keep_ids_num = 4096;
	m_keep_ids = (int *)malloc(sizeof(int)*m_keep_ids_num);

	m_BoxData_num = 4096;
	m_BoxArea_div = (float *)malloc(sizeof(float)*m_BoxData_num); 
	
}


int DetectionOutput::gen_anchors(void)
{
	int stride[3] = {8,16,32};
	int n = 0;
	m_anchor_num = 0;
	
	for (int i = 0;i < sizeof(stride)/sizeof(stride[0]);i++) {
		vector<int > f({m_in_height / stride[i],m_in_width /stride[i]});
		m_anchor_num += f[0] * f[1];
		m_feature_maps.push_back(f);
	}
	m_anchor_num *= 3;

	int anchor_size = m_anchor_num * 6;
	m_pAnchorBoxes = (float *)malloc(sizeof(float) * anchor_size);
	CHECK_EXPR(m_pAnchorBoxes == NULL,-1);
	float (*pBoxesArr)[6] = (float (*)[6])m_pAnchorBoxes;
	
    float anchors[][6] = {{10.,13., 16.,30., 33.,23.},
						 {30.,61., 62.,45., 59.,119.},
						 {116.,90., 156.,198., 373.,326.}};
	float (*pAnchorsArr)[3][2] = (float (*)[3][2])anchors;
	
	int idx = 0;
	for (int k = 0;k < m_feature_maps.size();k++) {
		int h = m_feature_maps[k][0];
		int w = m_feature_maps[k][1];
		
		for (int i = 0;i < h;i++) {
			for (int j = 0;j < w;j++) {
				for (int a = 0;a < 3;a++) {
					pBoxesArr[idx][0] = j;
					pBoxesArr[idx][1] = i;
					pBoxesArr[idx][2] = pAnchorsArr[k][a][0];
					pBoxesArr[idx][3] = pAnchorsArr[k][a][1];;
					pBoxesArr[idx][4] = stride[k];
					pBoxesArr[idx][5] = stride[k];
					idx++;
				}
			}
		}
	}
	
	return 0;
}


void DetectionOutput::nms_v0(vector<float  *> &pBoxes,vector<float * > &img_finnal_boxes)
{

	img_finnal_boxes.clear();
	float overlap = m_iou_thres;
	int num = pBoxes.size();
	CHECK_EXPR_NO_RETURN(num > m_keep_ids_num);
	int cnt = 0;
	for (int i = num-1;i >= 0;i--) {
		float area =  (pBoxes[i][2] -pBoxes[i][0])*(pBoxes[i][3] -pBoxes[i][1]);
		m_BoxArea_div[i] = 1./area;
		m_keep_ids[i] = 1;
	}
	for (int i = 0;i < num;i++ ) {
		if (m_keep_ids[i] == 0) {
			continue;
		}
		float xmin = pBoxes[i][0];
		float ymin = pBoxes[i][1];
		float xmax = pBoxes[i][2];
		float ymax = pBoxes[i][3];
		for (int j = i+1;j <num ;j++) {
			if (m_keep_ids[i] == 0) {
				continue;
			}
			float xx1 = (xmin > pBoxes[j][0]) ?xmin :pBoxes[j][0];
			float yy1 =  (ymin > pBoxes[j][1]) ?ymin :pBoxes[j][1];
			float xx2 =  (xmax < pBoxes[j][2]) ?xmax :pBoxes[j][2];
			float yy2 =  (ymax < pBoxes[j][3]) ?ymax :pBoxes[j][3];
			float w = xx2 - xx1;
			float h = yy2 - yy1;
			if (w > 0 && h > 0) {
				float o = w * h * m_BoxArea_div[i];
				if (o > overlap) {
					m_keep_ids[j] = 0;
				}  
			}
		}
	}
	for (int i = 0;i < num;i++) 
	{
		if (m_keep_ids[i] == 1) 
		{
			img_finnal_boxes.push_back(pBoxes[i]);
		}
	}
	return;
}


void DetectionOutput::print_bboxes(vector<float *> &boxes)
{
	// if(0 != boxes.size())
	// {
		for (int i = 0; i < boxes.size(); i++) 
		{
			std::cout << boxes[i][0] << " ";
			std::cout << boxes[i][1] << " ";
			std::cout << boxes[i][2] << " ";
			std::cout << boxes[i][3] << " ";
			std::cout << boxes[i][4] << " ";
			std::cout << boxes[i][5] << " ";	
			std::cout << std::endl;
		}        
	// }	
}


void DetectionOutput::Stage1(float *pred_in)
{
	float (*pPredInArr)[m_innersize] = (float (*)[m_innersize])pred_in;
	float (*pAnchorBoxesArr)[6] = (float (*)[6])m_pAnchorBoxes;
	float (*ppredFilter1Arr)[6] = (float (*)[6])m_predFilter1;
	mFilter1Num = 0;
	int idx = 0;
	//[cx cy w h , object_score, cls1_score, cls2_score,.....] ---> [xyxy, conf, cls_id]
	float cx,cy,w,h;
	for (int i = 0;i < m_anchor_num;i++) {

		if (pPredInArr[i][4] >= m_conf_thres) {
			
			float *pData = (float *)&pPredInArr[i][5];
			float obj_score = pPredInArr[i][4];
			int max_cls_id = -1;
			float max_score = 0.;
			for (int i = 0;i < m_num_cls;i++) {
				if (pData[i] >= m_conf_thres) {
					max_score = VOS_MAX(max_score,pData[i] * obj_score);
					max_cls_id = i;
				}
			}
			if (max_cls_id < 0 || max_score < m_conf_thres) {
				continue;
			}
			
			cx = (pPredInArr[i][0] * 2 - 0.5 + pAnchorBoxesArr[i][0]) * pAnchorBoxesArr[i][4];
			cy = (pPredInArr[i][1] * 2 - 0.5 + pAnchorBoxesArr[i][1]) * pAnchorBoxesArr[i][5];
			w =  (pPredInArr[i][2] * pPredInArr[i][2] * 4) * pAnchorBoxesArr[i][2];
			h =  (pPredInArr[i][3] * pPredInArr[i][3] * 4) * pAnchorBoxesArr[i][3];
			float w_div_2 = w / float(2);
			float h_div_2 = h / float(2);
			//CHECK_EXPR(idx > m_max_num1,finnal_boxes);
			if (idx >= m_max_num1) {
				break;
			}
			ppredFilter1Arr[idx][0] = cx - w_div_2;
			ppredFilter1Arr[idx][1] = cy - h_div_2;
			ppredFilter1Arr[idx][2] = cx + w_div_2;
			ppredFilter1Arr[idx][3] = cy + h_div_2;
			ppredFilter1Arr[idx][4] = max_score;
			ppredFilter1Arr[idx][5] = max_cls_id;

			m_cls_boxes[max_cls_id].push_back(ppredFilter1Arr[idx]);
			idx += 1;
		}
	}
	//cout << "predFilter1 number:" << idx << endl;
	//int size = idx * 6;
	//writeTxtFile("data_outputs/predFilter1.txt",m_predFilter1,size);
}


static bool comp(float *a, float *b) 
{
	return a[4] > b[4];
}


vector<float *>  &DetectionOutput::run(float *pred_in)
{
	m_output_finnal.clear();
	for (int c = 0;c < m_cls_boxes.size();c++) 
	{
		m_cls_boxes[c].clear();
		m_cls_boxes_finnal[c].clear();
	}
	Stage1(pred_in);
	for (int c = 0;c < m_cls_boxes.size();c++) 
	{
		if (m_cls_boxes[c].size() > 0) 
		{
			cout << "detection_output.cpp: DetectionOutput::run(): cls " << c << " num:" << m_cls_boxes[c].size()<<endl;
			sort(m_cls_boxes[c].begin(), m_cls_boxes[c].end(),comp);
			nms_v0(m_cls_boxes[c],m_cls_boxes_finnal[c]);
			for (int i = 0;i < m_cls_boxes_finnal[c].size();i++) 
			{
				m_output_finnal.push_back(m_cls_boxes_finnal[c][i]);
			}
		}
	}
	return m_output_finnal;
}






