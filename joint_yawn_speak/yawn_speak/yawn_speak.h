#ifndef YAWN_SPEAK_H
#define YAWN_SPEAK_H

#include "yawn_speak_global.h"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include "deque"
#include "math.h"
#include <iostream>
#include "datam_vless.h"
#define WIN 200
#define YAWN_win 33  //根据每秒帧数约定，约3s
#define YAWN_win_s 22  //根据每秒帧数约定，约2.5s
#define YAWN_s 0.51
#define YAWN 0.65
#define Yawn_up 7
#define least_time 22 //避免时间窗内嘴巴未还关上而导致的不报哈欠这一阈值即满足18帧就让哈欠+1
#define Pred_range 5 //丢失帧向前后搜索的帧数
#define alph_ave_l 0.15
#define alph_ave_r 0.25
#define alph_var_l 0.0015
#define alph_var_r 0.03
class YAWN_SPEAKSHARED_EXPORT Yawn_speak
{
public:
    int calc_alph(std::vector<cv::Point2d> &landmark, double &alph);
    int calc_pose(std::vector<cv::Point2d> &landmark,double &yaw1,double &pitch1,double &roll1,int col1,int row1);
    int calc_yawn(std::deque<DataM_vless> &data_q,int &result,int &ii);
    void quadraticSmooth7(std::deque<double> &in, std::deque<double> &out, int N);
};

#endif // YAWN_SPEAK_H
