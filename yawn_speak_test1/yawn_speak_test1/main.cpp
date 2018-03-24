#include <iostream>
#include <algorithm>
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include "string"
#include <sstream>
#include <cstdlib>
#include <math.h>
#include <deque>
#include <stdio.h>
#include <fstream>
#include <cstring>
#include <sstream>
#include <time.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/opencv.h>
#include <datam_vless.h>
#include <yawn_speak.h>
#define WIN_Total 720
//版本2
//每次测试需要修改两个地方：eye/yawn xm.avi xm.txt
using namespace  dlib;
bool cmp(int a,int b) {
    return a > b;
}
int pose(std::vector<cv::Point2d> &landmark, double &yaw1, double &pitch1, double &roll1, int col1, int row1, int &focus_state);
int main()
{
    FILE *fp;
    fp=fopen("/home/zzy/jilu_record/alph_update/6/1.txt","w");
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("/home/zzy/dlib-19.6/shape_predictor_68_face_landmarks.dat") >>pose_model;
    cv::VideoCapture capture(1);
 //   cv::VideoCapture capture("/home/zzy/jilu_record/0311/zzy_drive1.avi");
    std::deque<DataM_vless> data_q(200);
    std::deque<int> speaking_deque(720,0);
    std::deque<int> yawn_deque(720,0);
    std::deque<int> pose_deque(720,0);
    std::deque<int> yawn_deque_copy(720,0);
    cv::VideoWriter writer_color_raw;
    double rate=12;
    cv::Size colorvideoSize=cv::Size(640,480);

        /************write the video to avi*********/
//    if(!writer_color_raw.open("/home/zzy/jilu_record/0226/test_zzy.avi",CV_FOURCC('M','J', 'P', 'G'),rate, colorvideoSize))
//        { std::cout<<"writer open fail000"<<std::endl;
//            return 1;
//    }
/*********************开始打开摄像头**********************************/
    int ii=0;
    while(27!=cv::waitKey(30))

    {
    DataM_vless mouthtest;
    double yaw,pitch,roll;
    int a;
    Yawn_speak bb;
    cv::Mat frame;
    if(!capture.read(frame))break;

   // imshow("frame",frame);
    cv::Mat temp=frame.clone();

    /********************equalizeHist************/
    cv::Mat imageRGB[3];
        cv::split(temp, imageRGB);
        for (int i = 0; i < 3; i++)
        {
            cv::equalizeHist(imageRGB[i], imageRGB[i]);
        }
        cv::merge(imageRGB, 3, temp);
//    cv::cvtColor(temp,temp, cv::COLOR_BGR2GRAY);


//    int col1=frame.cols();
//    int row1=frame.rows();



    /*********************ert start**********************************/
    cv::medianBlur(temp,temp,3);
    cv::GaussianBlur(temp,temp,cv::Size(5,5),1);
    cv_image<bgr_pixel> cimg(temp);
   //calc the time :start
    std::clock_t start,finish;
    double face_dete_time;
    start=clock();

    std::vector<rectangle> faces = detector(cimg);

    //calc the time  finish
    finish=clock();
    face_dete_time=(double)(finish-start)/CLOCKS_PER_SEC;
    std::cout<<"face_detection_time:"<<face_dete_time<<"s"<<std::endl;


    //calc the time dlib_landmark
    start=clock();
    std::vector<full_object_detection> shapes;

    for (unsigned long j = 0; j < faces.size(); ++j)
    {
       // cv::rectangle(temp,dlibRectangleToOpenCV(faces[j]),cv::Scalar(0,0,128));
        shapes.push_back(pose_model(cimg, faces[j]));
    }
    /***********circle************/

     if (!shapes.empty()){
             for (int i = 0; i < shapes[0].num_parts(); i++) {
                 circle(frame, cvPoint(shapes[0].part(i).x(),
                        shapes[0].part(i).y()), 4, cv::Scalar(0, 0, 255), -1);
             }

    for (int j = 0; j <68;j++)
    {
         mouthtest.landmark.push_back(cv::Point2d(shapes[0].part(j).x(),(double)shapes[0].part(j).y()));
    }
    }

     int result,focus_state0;

     //calc the time landmark end
     finish=clock();
     double landmark_time=(double)(finish-start)/CLOCKS_PER_SEC;
     std::cout<<"alph_pose_time:"<<landmark_time<<"s"<<std::endl;


 //bb.calc_alph(mouthtest.landmark,mouthtest.alph);

     //calc the time of alph and pose
    start=clock();

    bb.calc_alph(mouthtest.landmark,mouthtest.alph);
    pose(mouthtest.landmark,yaw,pitch,roll,640,480,focus_state0);

        //calc the time of alph and pose end
    finish=clock();
    double alph_pose_time=(double)(finish-start)/CLOCKS_PER_SEC;
    std::cout<<"alph_pose_time:"<<alph_pose_time<<"s"<<std::endl;

    printf("%f\n",mouthtest.alph);

 //   printf("%d\n",data_q.size());
    if(data_q.size()<=WIN-1)
           data_q.push_back(mouthtest);
    else{
           data_q.pop_front();
           data_q.push_back(mouthtest);
    }

    //calc the time of  calc_yawn
     start=clock();

    int c=bb.calc_yawn(data_q,result,ii);

    finish=clock();
    double calc_yawn_time=(double)(finish-start)/CLOCKS_PER_SEC;
    std::cout<<"calc_yawn_time:"<<calc_yawn_time<<"s"<<std::endl;


    /********************calc the total score************************/
    int total_socre=0,speaking_frame=0,pose_frame=0,yawn_times=0,speak_degree=0,pose_degree=0,danger_degree=0;
    double speak_rate=0.0,pose_rate=0.0;
    if(speaking_deque.size()<=WIN_Total-1){
        if(result==6)speaking_deque.push_back(1);
        else speaking_deque.push_back(0);
        if(result>=1&&result<=4) {
            yawn_deque.push_back(result);
            yawn_deque_copy.push_back(result);
        }
        else  yawn_deque.push_back(0);

        if(focus_state0==1) pose_deque.push_back(1);
        else pose_deque.push_back(0);
    }

    else{
        speaking_deque.pop_front();
        yawn_deque.pop_front();
        pose_deque.pop_front();
        yawn_deque_copy.pop_front();
        if(result==6)speaking_deque.push_back(1);
        else speaking_deque.push_back(0);
        if(result>=1&&result<=4) {
            yawn_deque.push_back(result);
            yawn_deque_copy.push_back(result);
        }
        else
            {
            yawn_deque.push_back(0);
            yawn_deque_copy.push_back(0);
        }
        if(focus_state0==1) pose_deque.push_back(1);
        else pose_deque.push_back(0);
    }

    std::sort(yawn_deque_copy.begin(), yawn_deque_copy.end(),cmp);
    yawn_times=yawn_deque_copy[0];
    printf("yawn_times=%d\n",yawn_times);
    speaking_frame = accumulate(speaking_deque.begin() , speaking_deque.end() , 0);
    pose_frame = accumulate(pose_deque.begin() , pose_deque.end() , 0);
    speak_rate=1.0*speaking_frame/WIN_Total;
    pose_rate=1.0*pose_frame/WIN_Total;
    printf("%f %f %d\n",speak_rate,pose_rate,yawn_times);
    /**********************show the state**************/
    std::string danger_state1,danger_state2,danger_state3,danger="danger!";

    if(speak_rate>=0.5&&speak_rate<0.75)speak_degree=1;
    else if(speak_rate>=0.75)speak_degree=2;
    pose_degree=pose_rate/0.25;
    total_socre=50*speak_degree+30*yawn_times+50*pose_degree;
    double danger_coeffi= 0.05807*sqrt(total_socre);

    printf("total_score=%d\n",total_socre);
    if(total_socre>60){
        printf("Dangerous!\n");
        if(total_socre<100)danger_degree=1;
        if(total_socre>=100&&total_socre<150)danger_degree=2;
        if(total_socre>=150)danger_degree=3;
    }




    std::string text = "aph="+std::to_string(mouthtest.alph);
    int font_face = cv::FONT_HERSHEY_COMPLEX;
    double font_scale =0.5;
    int thickness = 1;
    //将文本框居中绘制
    cv::putText(frame, text, cv::Point(420,16), font_face, font_scale, cv::Scalar(0, 0,255), thickness, 3, 0);
    std::string C;
    if(result==6) C="speaking";
    if(result>=1&&result<=4) C="yawn";
    if(result==5||result==0) C="silence";
    if(result==100)C="attention off";
    std::string num="current state:"+C;
    cv::putText(frame, num, cv::Point(420,32), font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 3, 0);

    double font_scale1 =0.8;
//    std::string Title="Driver Behavior Detection System";

//    cv::putText(frame, Title, cv::Point(120,430), font_face, font_scale1, cv::Scalar(0, 255, 0), thickness, 3, 0);
     if(result>=1&&result<=4) {
         std::string times="times="+std::to_string(result);
         cv::putText(frame, times, cv::Point(420,45), font_face, 0.5, cv::Scalar(0, 0, 255), thickness, 3, 0);
     }

    /**********show the degree of pose**************/
    std::string str_yaw=std::to_string(yaw).substr(0,std::to_string(yaw).size()-4);
    std::string str_pitch=std::to_string(pitch).substr(0,std::to_string(pitch).size()-4);
    std::string str_roll=std::to_string(roll).substr(0,std::to_string(roll).size()-4);
    std::string str_focus;
    if(focus_state0==0)str_focus="ON";
    else str_focus="OFF";

    std::string Yaw="Yaw:"+str_yaw,Pitch="Pitch:"+str_pitch,Roll="Roll:"+str_roll;
    std::string Focus_state="Focus:"+str_focus;
    std::string dangerous_coeff="danger_coeffic:"+std::to_string(danger_coeffi);
    cv::putText(frame,Yaw,cv::Point(420,65), font_face, font_scale, cv::Scalar(0,0, 255), thickness, 3, 0);
    cv::putText(frame,Pitch,cv::Point(420,85), font_face, font_scale, cv::Scalar(0,0, 255), thickness, 3, 0);
    cv::putText(frame,Roll,cv::Point(420,105), font_face, font_scale, cv::Scalar(0,0, 255), thickness, 3, 0);
    cv::putText(frame,Focus_state,cv::Point(420,125), font_face, font_scale, cv::Scalar(0,0, 255), thickness, 3, 0);
    cv::putText(frame,dangerous_coeff,cv::Point(420,145), font_face, font_scale, cv::Scalar(0,0, 255), thickness, 3, 0);
    if(speak_rate>0.5){
        danger_state1="Long time speaking,"+danger;
        std::cout<<danger_state1<<std::endl;
        cv::putText(frame,danger_state1,cv::Point(200,420), font_face, font_scale, cv::Scalar(0,255, 0), thickness, 3, 0);
    }
    if(pose_rate>0.25) {
        danger_state2="Long time focus lost,"+danger;
       std::cout<<danger_state2<<std::endl;
       cv::putText(frame,danger_state2,cv::Point(200,440), font_face, font_scale, cv::Scalar(0,255, 0), thickness, 3, 0);

    }
    if(yawn_times>=2) {
        danger_state3="Fatigue detected,"+danger;
        std::cout<<danger_state3<<std::endl;
        cv::putText(frame,danger_state3,cv::Point(200,460), font_face, font_scale, cv::Scalar(0,255, 0), thickness, 3, 0);
    }

    cv::imshow("危险驾驶预警系统",frame);
     cv::imwrite("/home/zzy/jilu_record/alph_update/6/"+std::to_string(ii)+".jpg",frame);
 //   cv::imwrite("/home/zzy/jilu_record/0218/zzy_test_result_pic/"+std::to_string(ii)+".jpg",frame);
//     writer_color_raw<<frame;//write to avi
    printf("函数返回值为%d\n",c);
   fprintf(fp,"%f\t%f\t%f\t%f\t%d\n",mouthtest.alph,yaw,pitch,roll,result);
 //   fprintf(fp,"%f\t%f\t%f\t%f\n",face_dete_time,landmark_time,alph_pose_time,calc_yawn_time);//print out the time of 4 process
    printf("第%d帧\n",ii);
    ii++;
   }
}


int pose(std::vector<cv::Point2d> &landmark,double &yaw1,double &pitch1,double &roll1,int col1,int row1,int &focus_state)
{
    /*****************************pose estimation*******************************/
     if(!landmark.empty()){
    std::vector<cv::Point2d>image_points;
    image_points.push_back(cv::Point2d(landmark[30].x,landmark[30].y));    // Nose tip30
    image_points.push_back(cv::Point2d(landmark[8].x,landmark[8].y));    // Chin 8
    image_points.push_back(cv::Point2d(landmark[36].x,landmark[36].y));     // Left eye left corner 36
    image_points.push_back(cv::Point2d(landmark[45].x,landmark[45].y));    // Right eye right corner45
    image_points.push_back(cv::Point2d(landmark[48].x,landmark[48].y));    // Left Mouth corner 48
    image_points.push_back(cv::Point2d(landmark[54].x,landmark[54].y));    // Right mouth corner 54

    // 3D modelpoints.
    std::vector<cv::Point3d>model_points;
    model_points.push_back(cv::Point3d(0.0f,0.0f,0.0f));              // Nose tip
    model_points.push_back(cv::Point3d(0.0f,-330.0f, -65.0f));          //Chin
    model_points.push_back(cv::Point3d(-225.0f,170.0f, -135.0f));       // Left eye left corner
    model_points.push_back(cv::Point3d(225.0f,170.0f, -135.0f));        // Right eye rightcorner
    model_points.push_back(cv::Point3d(-150.0f,-150.0f, -125.0f));      // Left Mouth corner
    model_points.push_back(cv::Point3d(150.0f,-150.0f, -125.0f));       // Right mouth corner

   // Camerainternals
    double focal_length = col1; // Approximate focal length.
    cv::Point2d center =cv::Point2d(col1/2,row1/2);
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0,center.x, 0 , focal_length, center.y, 0, 0, 1);
    cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assumingno lens distortion
  //  std::cout <<"Camera Matrix " <<  std::endl << camera_matrix <<  std::endl ;

    // Outputrotation and translation
    cv::Mat rotation_vector; // Rotation in axis-angle form
    cv::Mat translation_vector;
    // Solve forpose
    cv::solvePnP(model_points,image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

    // Project a 3Dpoint (0, 0, 1000.0) onto the image plane.
    // We use thisto draw a line sticking out of the nose
    std::vector<cv::Point3d>nose_end_point3D;
    std::vector<cv::Point2d>nose_end_point2D;
    nose_end_point3D.push_back(cv::Point3d(0,0,1000.0));
    cv::projectPoints(nose_end_point3D,rotation_vector, translation_vector, camera_matrix, dist_coeffs,nose_end_point2D);
    cv::Mat R;
    cv::Rodrigues(rotation_vector,R);
    double q0=sqrt(double(1+R.at<double>(0,0)+R.at<double>(1,1)+R.at<double>(2,2)))/2;
    double q1=(R.at<double>(2,1)-R.at<double>(1,2))/(4*q0);
    double q2=(R.at<double>(0,2)-R.at<double>(2,0))/(4*q0);
    double q3=(R.at<double>(1,0)-R.at<double>(0,1))/(4*q0);
    double yaw=asin(double(2*(q0*q2 + q1*q3)));
    double pitch=atan2(double(2*(q0*q1-q2*q3)), double(q0*q0-q1*q1-q2*q2+q3*q3));
    double roll=atan2(2*(q0*q3-q1*q2), q0*q0+q1*q1-q2*q2-q3*q3);
    yaw1=yaw/3.14*180;
    pitch1=pitch/3.14*180-180;
    roll1=roll/3.14*180;
    if(pitch1 <= -180) pitch1 = pitch1 + 360;
    if(roll1 <= -90)roll1 = roll1 + 180;
    if(roll1 >= 90)roll1 = roll1 - 90;
     }
     else{

     yaw1=0.0,roll1=0.0,pitch1=0.0;

     }
     if(fabs(yaw1)>15||fabs(pitch1)>15||yaw1==0.0||roll1==0.0) focus_state=1;
     else focus_state=0;


     return 1;


}
