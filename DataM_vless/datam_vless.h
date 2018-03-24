#ifndef DATAM_VLESS_H
#define DATAM_VLESS_H

#include "datam_vless_global.h"
#include "iostream"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"

class DATAM_VLESSSHARED_EXPORT DataM_vless
{

public:
    DataM_vless();
    cv::Mat color_raw;
    cv::Mat depth_raw;
    cv::Mat color_result;
    cv::Mat depth_result;
    int frame_index;
    std::string color_raw_picname;
    std::string color_result_picname;
    std::string depth_raw_picname;

    std::string color_raw_videoname;
    std::string depth_raw_videoname;

    cv::Rect face; //face detected
    int face_flag; // front or profile or none
    std::vector<cv::Point2d> landmark; //ert calc
    std::vector<double> headpose; // head pose pnp
    std::vector<double> headpose_device; //recieved head pose through network
    double alph; // yawn calc
    double eye_percent; // eyes calc
    double nod_d1;      // nod calc
    double nod_d2;      // nod calc
    double nod_dist;
    double trans_z_abs;

    int alarm_status;

};

#endif // DATAM_VLESS_H
