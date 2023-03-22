#include"classify.h"


int Classify::init(CLA_CFG_t *pcfg)
{
    CHECK_EXPR(pcfg == NULL,-1);

    m_feed_height = pcfg->feed_h;
    m_feed_width = pcfg->feed_w;
    m_cls_num = pcfg->cls_num;    // Load RKNN Model

    int model_len = 0;
    bool exists = check_file_exists(pcfg->model);
    CHECK_EXPR(exists == false,-1);

    m_model = ENN::load_model(pcfg->model, &model_len);
    
    int ret = rknn_init(&m_ctx, m_model, model_len, 0);
    CHECK_EXPR(ret < 0,-1);

    ret = ENN::get_model_io_attrs(m_ctx,m_io_num,&m_p_input_attrs,&m_p_output_attrs);
    CHECK_EXPR(ret != 0,-1);

    /**********config output*************/
	m_outputs.want_float = 0;
	m_outputs.index = 0;
	m_outputs.is_prealloc = 1;
    int out_size = m_p_output_attrs[0].n_elems;

	m_outputs.buf = (unsigned char *)malloc(sizeof(unsigned char) * out_size);
	m_outputs.size = out_size * sizeof(unsigned char);

    return 0;
}

int Classify::run(unsigned char *p_feed_data, int& output)
{
 // Set Input Data,batch_size=1时，当batch_size=n，应该定义rknn_input inputs[n];
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = m_feed_height * m_feed_width * 3;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = p_feed_data;

    // cal_time_start();
    int ret = rknn_inputs_set(m_ctx, m_io_num.n_input, inputs);
    CHECK_EXPR(ret < 0,-1);
    // cal_time_end("==================rknn_inputs_set");
	
	int out_size = m_p_output_attrs[0].n_elems;
    //当m_outputs.buf通过调用rknn_outputs_get(m_ctx, 1, &m_outputs, NULL)函数获取到值时，ptr因为也指向这块内存，所以也有值了，后处理只需解析这段内存即可；
	uchar *ptr = (uchar *)m_outputs.buf;
    memset((unsigned char *)m_outputs.buf,0,out_size);

    Timer timer;
	for (int i = 0; i < 1; i++) 
    {
		// cal_time_start();
        timer.start();
		ret = rknn_run(m_ctx, nullptr);
		CHECK_EXPR(ret < 0,-1);
		// cal_time_end("rknn_run");

        
		// cal_time_start();
		ret = rknn_outputs_get(m_ctx, 1, &m_outputs, NULL);
		CHECK_EXPR(ret < 0,-1);
		// cal_time_end("rknn_outputs_get");
        std::cout<<"Outputs res="<<int(ptr[0]);

		// Post Process，实现argMax函数(针对单张图)
        float val_max = ptr[0];
        unsigned char arg_max = 0;
        for (int c = 1; c < m_cls_num; ++c) 
        {

            std::cout<<"  "<<int(ptr[c])<<std::endl;
            if (ptr[c] > val_max) 
            {
                arg_max = c;
                val_max = ptr[c];
            }
        }
        output = arg_max;
    }
    return 0;
}



Classify::~Classify()
{
    if(m_ctx >= 0) 
    {
        rknn_destroy(m_ctx);
    }
    
    if(m_model) 
    {
        free(m_model);
    }

}
