#include "yawn_speak.h"
#include "math.h"
#include "deque"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include "string.h"
#include <iostream>
#include <datam_vless.h>
using namespace std;

int Yawn_speak::calc_alph(std::vector<cv::Point2d> &landmark,double &alph)
{

    //函数用于计算每一帧所得到的张开度alph值
    if(!landmark.empty()){
        /***************************求60-67号形状的外接矩形高宽比,如没有landmark，alph值记为0*********************************/
       std::vector<cv::Point2i> contours;
       for(int mm=60;mm<68;mm++)
          contours.push_back(cvPoint(landmark[mm].x,landmark[mm].y));//landmark[index]
       cv::RotatedRect rect=minAreaRect(contours);
       cv::Point2f P[4];
       rect.points(P);
       double ww=sqrt((P[0].x-P[1].x)*(P[0].x-P[1].x)+(P[0].y-P[1].y)*(P[0].y-P[1].y));
       double hh=sqrt((P[2].x-P[1].x)*(P[2].x-P[1].x)+(P[2].y-P[1].y)*(P[2].y-P[1].y));
       if(ww<hh)alph=ww/hh;
       else alph=hh/ww;


    }
    else alph=0.0;
    return 1;
}
int Yawn_speak::calc_pose(std::vector<cv::Point2d> &landmark,double &yaw1,double &pitch1,double &roll1,int col1,int row1)
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
     return 1;


}
int Yawn_speak::calc_yawn(std::deque<DataM_vless> &data_q, int &result1,int &ii)
{
     //输入是data_q，result计算结果，ii（为了方便测试传入传出当前帧数）为当前帧数变量，ii可去除,同时需在.h文件中去掉。

    //函数用于判断是否发生哈欠 result表示发生哈欠的次数，其中变量yawn_state2表示深哈欠次数，yawn_state1浅哈欠次数
    std::deque<int> state_mouth(WIN,0); //设定滑动窗口大小,初始化为5
    std::deque<double> deque_mouth(WIN,0.0);
    std::deque<double> deque_mouth_fit(WIN,0.0);
    int yawn_state[10]={0};
    int yawn=0;//control the up time
    int result=0;
    result1=0;
    int times=0,j=0,flag=0;//times为窗口内哈欠次数，j为一般循环变量，jj用于控制记录哈欠的数组

    static int tt=0;//tt保持当前连续超过阈值YAWN_s的时间帧
    int max_time=0;//保存最长哈欠帧数
    int yawn_time[10]={0};   //数组保存每次发生哈欠的连续帧数，数组非0数字的个数即为哈欠次数，滑动窗口内最多打10次哈欠（可用于哈欠最长帧计算）
    int flag_null=0,count0=0; //flag_null,count0用于辅助判断连续5帧不存在脸的情况（5帧可修改）
    double alph_sum=0.0,alph_max=0.0,alph_ave=0.0,alph_var=0.0;
    int yawn_flag=0;//statistic judge 1
    /**********将data_q中的压入函数中的deque,便于函数中统计计算，新的值压入队尾*************/
    if(data_q.size()>=WIN)flag=1;
    if(flag==1){
        for(int i=0;i<WIN;i++)
        {
            if(deque_mouth.size()<=WIN-1)
            {
                deque_mouth.push_back(data_q[data_q.size()-1-i].alph);

            }
            else
            {
                deque_mouth.pop_front();
                deque_mouth.push_back(data_q[data_q.size()-1-i].alph);
            }

        }
      /**********state_mouth[]为0-1序列，deque_mouth大于阈值为1**********************/

        for(int i=0;i<WIN;i++)
        {
        if(deque_mouth[i]>YAWN_s)
            state_mouth[i]=1;
        else state_mouth[i]=0;
        }

        /****************************显示点的坐标************************************/

        cv::Mat axis(200,WIN,CV_32F);

        printf("\n");
        for(int i=0;i<200;i++)
            for(int j=0;j<WIN;j++)
                     axis.at<float>(i,j)=0;
        for(int k=0;k<WIN;k++)
        cv::circle(axis, cvPoint(k,200-deque_mouth[k]*100), 2, cv::Scalar(255, 0, 0), -1);
        cv::imshow("mouth_open", axis);


        /****************************处理数据，未检测到人脸的情况！************************************/
        for(int i=0;i<5;i++)
        {
        if(data_q[data_q.size()-1-i].alph==0.0){
            count0++;
            if(flag_null==0) flag_null=1;
            if(count0>=5){
                    result1=100;   //隐去返回值,return后就退出函数不执行后续程序.

                    count0=0;
                    printf("当前连续帧未检测到人脸！\n");
                    flag_null=0;
                    return result1;  //attention off
            }
        }
        }


        /******************************开始处理数据*********************************/

        int kk=0,k=0,sum=0;float sum_alph=0.0;
        /******************************处理异常帧********************************/

    /*Pred_range定义了搜索的左边界和右边界的范围,可在.h文件中修改，在丢失帧的前后范围内找大于阈值的alph并求平均值，若无大于阈值的仍为0*/
        for(k=Pred_range;k<=WIN-Pred_range;k++){
            if(deque_mouth[k]==0.0){
                for(kk=k+Pred_range;kk>=k-Pred_range;kk--){
                     {
                         sum=sum+state_mouth[kk];
                         if(state_mouth[kk]==1)sum_alph=sum_alph+deque_mouth[kk];   //
                     }
                }
                if(sum>=1)deque_mouth[k]=sum_alph/(sum*1.0);
                if(deque_mouth[k]>YAWN_s)state_mouth[k]=1;
                sum=0;sum_alph=0.0;
            }

        }
        sum=0;sum_alph=0.0;
        for(k=WIN-(Pred_range-1);k<=WIN-1;k++){
            if(deque_mouth[k]==0.0){
                for(kk=k-(Pred_range-1);kk>=k-1;kk++){
                     {
                         sum=sum+state_mouth[kk];
                         if(state_mouth[kk]==1)sum_alph=sum_alph+deque_mouth[kk];   //
                     }
                }
                if(sum>=1)deque_mouth[k]=sum_alph/(sum*1.0);
                if(deque_mouth[k]>YAWN_s)state_mouth[k]=1;
                sum=0;sum_alph=0.0;

            }

        }


        sum=0;sum_alph=0.0;
        /***********再扫描一次，针对101的情况把丢失的中间帧补上***********************/
        for(k=1;k<=WIN-2;k++){
            if(state_mouth[k]==0){            //改为deque_mouth
            if(state_mouth[k-1]==1&&state_mouth[k+1]==1)
                       {state_mouth[k]=1;
                        deque_mouth[k]=(deque_mouth[k-1]+deque_mouth[k+1])*0.5;}

            }
        }
      /***************************显示第一次预处理后的数据************************************/
        cv::Mat axis1(200,WIN,CV_32F);
        for(int i=0;i<200;i++)
            for(int j=0;j<WIN;j++)
                     axis1.at<float>(i,j)=0;
        for(int k=0;k<WIN;k++)
        cv::circle(axis1, cvPoint(k,200-deque_mouth[k]*100), 2, cv::Scalar(255, 0, 0), -1);
        cv::imshow("mouth_open1", axis1);

      /***************************第二次平滑处理后的数据并显示************************************/
        quadraticSmooth7(deque_mouth, deque_mouth_fit, WIN);  //送入函数


        cv::Mat axis2(200,WIN,CV_32F);
        for(int i=0;i<200;i++)
            for(int j=0;j<WIN;j++)
                     axis2.at<float>(i,j)=0;
        for(int k=0;k<WIN;k++)
        cv::circle(axis2, cvPoint(k,200-deque_mouth_fit[k]*100), 2, cv::Scalar(255, 0, 0), -1);
        cv::imshow("mouth_open2", axis2);

      /*************calculate the average and variance*********************/
        int jj=0;
        for(int ii=0;ii<WIN-1;ii++)
            {  if(deque_mouth_fit[ii]!=0)jj++;
               alph_sum=deque_mouth_fit[ii]+alph_sum;
               if(deque_mouth[ii]>alph_max)alph_max=deque_mouth[ii];
            }
        alph_ave=1.0*alph_sum/jj;
        for(int ii=0;ii<WIN-1;ii++)
        {
            if(deque_mouth[ii]!=0)alph_var=alph_var+(deque_mouth[ii]-alph_ave)*(deque_mouth[ii]-alph_ave);
        }
        alph_var=1.0*alph_var/jj;
        printf("alph_ave=%f alph_var=%f,alph_max=%f\n",alph_ave,alph_var,alph_max);

         if(alph_ave<0.18&&alph_var<0.002&&alph_max<0.5) result=5;//5=silence
          if(alph_ave>0.18&&alph_var>0.04&&alph_max>0.65) yawn_flag=1;//1=yawn
        if(yawn_flag!=1&&alph_ave>alph_ave_l&&alph_ave<alph_ave_r&&alph_var>alph_var_l &&alph_var<alph_var_r&&alph_max<0.5) result=6;//6=speak
        //alph_ave_l、alph_ave_r、alph_var_l、alph_var_r为说话检测时alph时间序列的均值、方差左右约束，可在.h文件中进行修改。



        /*******************************计算哈欠次数和对应时间，时间窗从前到后遍历，碰到一次哈欠立即保存至数组*********************************/
        int count1=0,count2=0,count3=0; int count_up=0,flag_up=0;
        jj=0;
        for(k=WIN-1;k>=0;k--)
        {
            if(deque_mouth_fit[k]>0.2&&deque_mouth_fit[k-1]>deque_mouth_fit[k]){
                flag_up=1;
                if(flag_up==1)count_up++;
                if(count_up>=Yawn_up){yawn=1;} //单调连续递增阈值
             }
            else
                {
                flag_up=0;
                count_up=0;
                if(deque_mouth_fit[k]<0.2){yawn=0;   //alph值掉下来，则上升趋势yawn置0，说明后续不再具有哈欠特征
                   }
                }
  //          printf("up=%d\nyawn=%d\n",count_up,yawn);//输出alph上升阶段时间及yawn变量的状态

            if(state_mouth[k]==1)//统计保持大于阈值YAWN_s和YAWN的帧数
            {flag=1;
             if(deque_mouth_fit[k]>=YAWN_s){tt++;count1++;}  // count1为小阈值
             if(deque_mouth_fit[k]>=YAWN_s&&deque_mouth_fit[k]<=YAWN)count2++;//count2为介于小阈值与大阈值间
             if(deque_mouth_fit[k]>YAWN)count3++;//count3为大于较大阈值的帧数
             if(count1>=YAWN_win_s)yawn=1;// yawn=0!若count1大于较小时间窗阈值,yawn保持为1，保证上个时间窗残留哈欠可以正常通过下个时间窗
            }



    //       printf("flag=%d,yawn=%d,count1=%d count2=%d count3=%d\n",flag,yawn,count1,count2,count3);

     /**********************嘴巴张开度掉下来后保持标志哈欠停止，记录本次哈欠，flag为连续张大的标志，yawn为本次张嘴过程达到持续上升时间阈值的标志，两者同时为1，且持续张大时间满足阈值可判断为哈欠****************/
            if(state_mouth[k]==0){

                           if(flag==1){

                           tt=0;
                          if(yawn==1){
                               if(count1>=YAWN_win_s)  //设定满足哈欠的帧数阈值YAWN_win
                               {
                                   if(count3>=YAWN_win){
                                       yawn_state[jj++]=2;
                                    //  printf("一次深哈欠!\n");//当且仅当张开度较大、持续时间较长才判为深哈欠
                                   }
                                   else {

                                       yawn_state[jj++]=1;
                                      // printf("一次浅哈欠!\n");
                                   }
                                   yawn_time[j++]=count1;
                                 //  printf("flag=%d,yawn=%d,count1=%d count2=%d count3=%d\n",flag,yawn,count1,count2,count3);

                               }
                           }
                          //完成一次记录后清0，进行下一阶段判断
                          flag=0;
                          yawn=0;
                          count1=0;count2=0;count3=0;


                   }

                   flag=0;tt=0;count1=0;count2=0;count3=0;
            }

        }

       /****************深哈欠和浅哈欠的次数,及最长哈欠持续时间********************/

        int yawn_state1=0,yawn_state2=0; //浅，深
        for(k=0;k<j;k++)  //计算持续哈欠的最大帧数q
          {
              if(yawn_state[k]==2) yawn_state2++; //输出深哈欠次数
              if(yawn_state[k]==1) yawn_state1++; //输出浅哈欠次数
              if(max_time<yawn_time[k])max_time=yawn_time[k];
          }
        if(tt>=least_time) {  //在时间窗末端发生的哈欠可能由于嘴巴未合上而未能统计，此处增加当前连续帧大于least_time阈值，则报告哈欠，并记录为一次浅哈欠
            times=yawn_state2+yawn_state1+1;
        yawn_state1++;//  add   !!!
        if(tt>max_time) max_time=tt; //add!!!
            tt=0;
        }//设定满足哈欠的帧数阈值,当没有转换时对此进行判断次数保存为j
        else
        {
            times=yawn_state2+yawn_state1;
        }

        if(yawn_state1!=0||yawn_state2!=0){
              printf("打哈欠%d次 %d次深哈欠，%d次浅哈欠\n",times,yawn_state2,yawn_state1);
                  printf("longest  yawn time %d frames\n",max_time);//add!!!
          }
        for(k=0;k<10;k++)   //归0 最多10次哈欠
             yawn_time[k]=0;
        count_up=0;
    }//用于统计的for循环的末括号
        j=0;tt=0;//j记录当前窗口次数
        if(yawn_flag==1)result1=times;
        else result1=result;
        times=0;
        printf("result=%d\n",result);

        return result;



 //用times哈欠次数，yawn_state2保存深哈欠次数，yawn_state1保存浅哈欠次数，max_time最长哈欠帧数，result为当前时间窗哈欠次数 (未检测到脸暂仍返回0，可在程序中修改；有哈欠为哈欠次数，无则为0）
  //result=100 attention out  result=5 silence result=6 speaking  result 1-4 yawn times


}
void Yawn_speak::quadraticSmooth7(std::deque<double> &in, std::deque<double> &out, int N)
{
     //7点二次函数平滑函数,以处理数据抖动带来的误差，曲线向中间收敛
      int i;
      if ( N < 7 )
      {
          for ( i = 0; i <= N - 1; i++ )
          {
              out[i] = in[i];
          }
      }
      else
      {
          out[0] = ( 32.0 * in[0] + 15.0 * in[1] + 3.0 * in[2] - 4.0 * in[3] -
                    6.0 * in[4] - 3.0 * in[5] + 5.0 * in[6] ) / 42.0;

          out[1] = ( 5.0 * in[0] + 4.0 * in[1] + 3.0 * in[2] + 2.0 * in[3] +
                    in[4] - in[6] ) / 14.0;

          out[2] = ( 1.0 * in[0] + 3.0 * in [1] + 4.0 * in[2] + 4.0 * in[3] +
                    3.0 * in[4] + 1.0 * in[5] - 2.0 * in[6] ) / 14.0;
          for ( i = 3; i <= N - 4; i++ )
          {
              out[i] = ( -2.0 * (in[i - 3] + in[i + 3]) +
                         3.0 * (in[i - 2] + in[i + 2]) +
                        6.0 * (in[i - 1] + in[i + 1]) + 7.0 * in[i] ) / 21.0;
          }
          out[N - 3] = ( 1.0 * in[N - 1] + 3.0 * in [N - 2] + 4.0 * in[N - 3] +
                        4.0 * in[N - 4] + 3.0 * in[N - 5] + 1.0 * in[N - 6] - 2.0 * in[N - 7] ) / 14.0;

          out[N - 2] = ( 5.0 * in[N - 1] + 4.0 * in[N - 2] + 3.0 * in[N - 3] +
                        2.0 * in[N - 4] + in[N - 5] - in[N - 7] ) / 14.0;

          out[N - 1] = ( 32.0 * in[N - 1] + 15.0 * in[N - 2] + 3.0 * in[N - 3] -
                        4.0 * in[N - 4] - 6.0 * in[N - 5] - 3.0 * in[N - 6] + 5.0 * in[N - 7] ) / 42.0;
      }
      for(int i=0;i<N;i++)
          if(out[i]<0)out[i]=fabs(out[i]);

}
