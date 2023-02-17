// ==========================================================
// 实现功能：使用OpenVINO推理unetpp，将所有操作封装成加载模型、推理、释放，隐藏OpenVINO相关代码；
// 文件名称：ovInferenceUnetpp.cpp
// 相关文件：无
// 作   者：Liangliang Bai (liangliang.bai@leapting.com)
// 版   权：<Copyright(C) 2022-Leapting Technology Co.,LTD All rights reserved.>
// 修改记录：
// 日   期              版本       修改人      走读人      修改记录
// 2023.02.06   1.0.0.1   白亮亮                          None
// ==========================================================

#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include "format_reader_ptr.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
// clang-format on

#ifndef INPUT
#define INPUT
#endif

#ifndef OUTPUT
#define OUTPUT
#endif


// 实现功能：Print algoritmn version；
// ----------------------------------->parameters<----------------------------------
// inputParas :
//     None
// outputParas:
//     None
// returnValue:None;
// ----------------------------------->parameters<----------------------------------
void getVersion()
{
    const char* algVersion =" v1.00.000";
    std::cout<<"Algorithmn version is:"<<algVersion<<std::endl;
    slog::info << ov::get_openvino_version() << slog::endl;
}

/**
 * @brief Main with support Unicode paths, wide strings
 */
int main() 
{
    getVersion();
    try 
    {
        const std::string model_path = "/home/bailiangliang/OPENVINO_TEST/model/416x512.onnx";
        const std::string image_path = "/home/bailiangliang/OPENVINO_TEST/test_image/1/Color_1.bmp";
        const std::string device_name = "CPU";

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::Core core;

        // -------- Step 2. Read a model --------
        slog::info << "Loading model files: " << model_path << slog::endl;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        printInputAndOutputsInfo(*model);

        // 当前仅支持单输入和单输出头
        OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
        OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

        // -------- Step 3. Set up input
        // Read input image to a tensor and set it to an infer request
        // without resize and layout conversions
        FormatReader::ReaderPtr reader(image_path.c_str());
        if (reader.get() == nullptr) 
        {
            std::stringstream ss;
            ss << "Image " + image_path + " cannot be read!";
            throw std::logic_error(ss.str());
        }

        ov::element::Type input_type = ov::element::u8;
        std::cout<<"reader->height()="<<reader->height()<<"; reader->width()="<<reader->width()<<std::endl;
        ov::Shape input_shape = {1, reader->height(), reader->width(), 3};
        std::shared_ptr<unsigned char> input_data = reader->getData();  // 读入图片，返回一个指针
        
        // just wrap image data by ov::Tensor without allocating of new memory
        ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, input_data.get());
        const ov::Layout tensor_layout{"NHWC"};

        // -------- Step 4. Configure preprocessing --------
        ov::preprocess::PrePostProcessor ppp(model);

        // 1) Set input tensor information:
        // - input() provides information about a single model input
        // - reuse precision and shape from already available `input_tensor`
        // - layout of data is 'NHWC'
        ppp.input().tensor().set_shape(input_shape).set_element_type(input_type).set_layout(tensor_layout);
        // ppp.input().preprocess().convert_color(ov::preprocess::ColorFormat::RGB);  // 根据需要转换为 RGB 格式.
        
        // 要把输入改为 f32 类型，必须用 convert_element_type，而不能用 ppp.input().tensor().set_element_type，否则无效。
        ppp.input().preprocess().convert_element_type(ov::element::f32);
        ppp.input().preprocess().scale(255.0f);  // 做归一化处理，除以 255
        
        

        // 2) Adding explicit preprocessing steps:
        // - convert layout to 'NCHW' (from 'NHWC' specified above at tensor layout)
        // - apply linear resize from tensor spatial dims to model spatial dims
        ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);

        // 4) Here we suppose model has 'NCHW' layout for input
        ppp.input().model().set_layout("NCHW");

        // 5) Set output tensor information:
        // - precision of tensor is supposed to be 'f32'
        ppp.output().tensor().set_element_type(ov::element::f32);

        // 6) Apply preprocessing modifying the original 'model'
        model = ppp.build();

        // -------- Step 5. Loading a model to the device --------
        ov::CompiledModel compiled_model = core.compile_model(model, device_name);

        // -------- Step 6. Create an infer request --------
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        // -----------------------------------------------------------------------------------------------------

        // -------- Step 7. Prepare input --------
        infer_request.set_input_tensor(input_tensor);

        // -------- Step 8. Do inference synchronously --------
        infer_request.infer();

        // -------- Step 9. Process output
        const ov::Tensor& output_tensor = infer_request.get_output_tensor();
        std::cout<<"output_tensor="<<output_tensor.data()<<std::endl;
        float * output = (float*)output_tensor.data();
        
        cv::Mat m_seg_out = cv::Mat::zeros(416, 512, CV_8UC1);
        for (int h = 0;h < 416;h++) 
        {
            for (int w = 0;w < 512 ;w++) 
            {
                float val_max = output[h*512+w];
                unsigned char arg_max = 0;
                for (int c = 1;c < 3; c++) 
                {
                    if (output[h*512+w+512*416*c] > val_max)
                    {
                        arg_max = c;                                                                                                                                                                              //更新arg_max
                        val_max = output[h*512+w+512*416*c];                //更新当前最大值
                    }
                }
                m_seg_out.at<uchar>(h,w) = int(arg_max)*60;
            }
	    }

        cv::imwrite("/home/bailiangliang/OPENVINO_TEST/image_test_result/xxxxxxx.png", m_seg_out);


        //5、图像后处理，并将后处理图映射回原图尺寸
		// cv::Mat singleSegOut;
        // cv::Mat orgImg = cv::imread(image_path);
        // cv::Mat colorImg;
        // std::string imgSavePath = "~/inference_solar_panel/inference_result.bmp";

        // imagePostprocess(unetppTensorParams, output, singleSegOut);

        // cv::resize(singleSegOut, singleSegOut, cv::Size(orgImg.cols, orgImg.rows), (0, 0), (0, 0), cv::INTER_LINEAR);
        // std::cout << "debug 3, before singleChannel2threeeChannel" << std::endl;
        // // singleChannel2threeeChannel(singleSegOut, colorImg);    
        
        // cv::imwrite(imgSavePath, colorImg);
        
        // cv::Mat input_image = cv::imread(image_path, cv::IMREAD_COLOR);
        // cv::imshow("input_image", input_image);

        // Print classification results
        ClassificationResult classification_result(output_tensor, {image_path});
        classification_result.show();
        // ---------------------------- -------------------------------------------------------------------------
    } 
    catch (const std::exception& ex) 
    {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


// ./openvinoTest /home/bailiangliang/OPENVINO_TEST/model/416x512.onnx /home/bailiangliang/OPENVINO_TEST/test_image/1/Color_1.bmp CPU


