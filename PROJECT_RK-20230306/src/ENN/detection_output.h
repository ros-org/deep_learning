#ifndef __DETECTION_OUTPUT_H__
#define __DETECTION_OUTPUT_H__

#include "common.h"

class DetectionOutput
{
public:
	DetectionOutput(){};
	~DetectionOutput(){};
	void init(int ori_width,int ori_height,int num_cls);
	vector<float *>  &run(float *pred_in);
	void  print_bboxes(vector<float *> &boxes);
	
protected:

	int gen_anchors(void);
	void Stage1(float *pred_in);
	void nms_v0(vector<float  *> &pBoxes,vector<float * > &img_finnal_boxes);
		
	Dtype *m_priorBoxes;
	Dtype *m_predFilter1;
	int mFilter1Num;

	int m_in_width;
	int m_in_height;
	int m_num_cls;
	int m_anchor_num;
	int m_max_num1;//after filter by object score
	int m_innersize;
	float m_conf_thres;
	float m_iou_thres;
	
	vector<int > m_idxs;
	//vector<vector<float > > m_scores;

	vector<vector<int> > m_feature_maps;
	float *m_pAnchorBoxes;

	vector<float * > m_finnal_boxes;
	vector<vector<float *> > m_cls_boxes;
	vector<vector<float *> > m_cls_boxes_finnal;

	vector<float *>  m_output_finnal;
	
	int m_keep_ids_num;
	int *m_keep_ids;

	int m_BoxData_num;
	float *m_BoxArea_div;
};


#endif



