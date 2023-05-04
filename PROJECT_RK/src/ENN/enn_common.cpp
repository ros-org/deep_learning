#include "enn_common.hpp"

unsigned char *ENN::load_model(char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if(fp) {
        fclose(fp);
    }
    return model;
}


void ENN::printRKNNTensor(rknn_tensor_attr *attr) 
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n", 
            attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0], 
            attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}


int ENN::get_model_io_attrs(rknn_context &ctx,
                            rknn_input_output_num &io_num,
                            rknn_tensor_attr **input_attrs,
                            rknn_tensor_attr **output_attrs)
{
    // Get Model Input Output Info
    int ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");

    rknn_tensor_attr * p_input_attrs = (rknn_tensor_attr *)malloc(sizeof(rknn_tensor_attr) * io_num.n_input);
    CHECK_EXPR(p_input_attrs == NULL,-1);

    memset(p_input_attrs, 0, sizeof(rknn_tensor_attr) * io_num.n_input);
    for (int i = 0; i < io_num.n_input; i++) 
    {
        p_input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(p_input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        ENN::printRKNNTensor(&(p_input_attrs[i]));
    }
    *input_attrs = p_input_attrs;

    printf("output tensors:\n");

    rknn_tensor_attr *p_output_attrs = (rknn_tensor_attr *)malloc(sizeof(rknn_tensor_attr) * io_num.n_output);
    CHECK_EXPR(p_output_attrs == NULL,-1);

    memset(p_output_attrs, 0, sizeof(rknn_tensor_attr)* io_num.n_output);
    for (int i = 0; i < io_num.n_output; i++) 
    {
        p_output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(p_output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) 
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        ENN::printRKNNTensor(&(p_output_attrs[i]));
    }
    *output_attrs = p_output_attrs;

    return 0;
} 
