#include "Segmentation.hpp"
#include "enn_common.hpp"

int Segmentation::init(SEG_CFG_t *pcfg)
{
    int model_len = 0;
    m_Cfg = *pcfg;

    bool exists = check_file_exists(pcfg->model);
    CHECK_EXPR(exists == false,-1);

    m_model = ENN::load_model(pcfg->model, &model_len);

    int ret = rknn_init(&m_ctx, m_model, model_len, 0);
    CHECK_EXPR(ret != 0,-1);

    ret = ENN::get_model_io_attrs(m_ctx,m_io_num,&m_p_input_attrs,&m_p_output_attrs);
    CHECK_EXPR(ret != 0,-1);

     /**********config output*************/
	m_outputs.want_float = 1;
	m_outputs.index = 0;
	m_outputs.is_prealloc = 1;
	int out_size = m_p_output_attrs[0].n_elems;

    cout << "out_size:" << out_size << endl;
	m_outputs.buf = (float *)malloc(sizeof(float) * out_size);
	m_outputs.size = out_size * sizeof(float);

    // m_p_seg_data = (unsigned char *)malloc(sizeof(unsigned char) * m_Cfg.feed_h * m_Cfg.feed_w);
    m_seg_out = Mat::zeros(m_Cfg.feed_h, m_Cfg.feed_w, CV_8UC1);
    return 0;
}

int Segmentation::run(unsigned char *p_feed_data,Mat &im_seg)
{
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = m_Cfg.feed_h * m_Cfg.feed_w * 3;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = p_feed_data;

    int ret = rknn_inputs_set(m_ctx, m_io_num.n_input, inputs);
    if(ret < 0) 
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }
    int cls_num = m_Cfg.cls_num;
    int feed_h  = m_Cfg.feed_h;
    int feed_w  = m_Cfg.feed_w;
    // cout << "Segmentation::run feed_h:" << feed_h << endl;
    // cout << "Segmentation::run feed_w:" << feed_w << endl;
    // cout << "Segmentation::run cls_num:" << cls_num << endl;

    float (*ptr_out)[feed_h][feed_w] = (float (*)[feed_h][feed_w])m_outputs.buf;
    unsigned char (*ptr_out_decode)[feed_w] = (unsigned char (*)[feed_w])m_seg_out.data;//(unsigned char (*)[feed_w])m_seg_out.data;
    int out_size = m_p_output_attrs[0].n_elems;
    Timer timer;
    int test_cnt = 1;
    for (int i = 0;i < test_cnt;i++) {

        timer.start();
        ret = rknn_run(m_ctx, nullptr);
        CHECK_EXPR(ret != 0,-1);

        ret = rknn_outputs_get(m_ctx, 1, &m_outputs, NULL);
        CHECK_EXPR(ret != 0,-1);

        #if 1
        for (int h = 0;h < feed_h;h++) {
            for (int w = 0;w < feed_w;w++) 
            {
                float val_max = ptr_out[0][h][w];
                unsigned char arg_max = 0;
                for (int c = 1;c < cls_num;c++) 
                {
                    if (ptr_out[c][h][w] > val_max) 
                    {
                        arg_max = c;
                        val_max = ptr_out[c][h][w];
                    }
                }
                ptr_out_decode[h][w] = arg_max;
            }
        }
        #endif
        
        timer.end("Seg inference ..");
    }

    ENN::printRKNNTensor(&m_p_output_attrs[0]);

    im_seg = m_seg_out;
    return 0;
}

void Segmentation::draw_seg(Mat &img, Mat &im_seg)
{   
    int feed_h = m_Cfg.feed_h;
    int feed_w = m_Cfg.feed_w;
    int cls_num = m_Cfg.cls_num;

    unsigned char (*p_src)[feed_w] = (unsigned char (*)[feed_w])im_seg.data; // m_seg_out.data;
    unsigned char (*p_dst)[feed_w][3] = (unsigned char (*)[feed_w][3])img.data;

    unsigned char colors[][3] = {{0,0,0},{0,0,255},{255,0,0},{0,255,0},{255,255,0},{255,0,255},{0,255,255}};
    ASSERT(cls_num <= DIM_OF(colors));

    for (int h = 0;h < feed_h;h++) {
        for (int w = 0;w < feed_w;w++) {
            int cls = p_src[h][w];
            if (cls > 0) {
                for (int c = 0;c < 3;c++) {
                    p_dst[h][w][c] = colors[cls][c];
                }
            }
        }
    }
}


int Segmentation::post_process(Mat &im_seg)
{

    return 0;
}

Segmentation::~Segmentation()
{
    
    if(m_ctx >= 0) {
        rknn_destroy(m_ctx);
    }
}