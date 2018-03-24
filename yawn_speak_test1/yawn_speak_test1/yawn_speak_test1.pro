#-------------------------------------------------
#
# Project created by QtCreator 2018-01-29T16:03:06
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = yawn_speak_test1
CONFIG   += console
CONFIG   -= app_bundle
CONFIG   += c++11
TEMPLATE = app


SOURCES += main.cpp
DEFINES += QT_DEPRECATED_WARNINGS
INCLUDEPATH += /home/zzy/QT_app/Data_struct
INCLUDEPATH += /usr/local/include/opencv
INCLUDEPATH += /usr/local/include/opencv2
INCLUDEPATH += /home/zzy/dlib-19.6
INCLUDEPATH += /home/zzy/tbb/include
INCLUDEPATH += /home/zzy/QT_app/joint_yawn_speak/yawn_speak
INCLUDEPATH += /home/zzy/QT_app/DataM_vless

LIBS += -L/usr/local/lib\
    -lopencv_core\
    -lopencv_imgproc\
    -lopencv_highgui\
    -lopencv_features2d\
    -lopencv_calib3d\
    -lopencv_imgcodecs\
    -lopencv_videoio\
    -lopencv_flann\
    -lopencv_ml\
    -lopencv_objdetect\
    -lopencv_photo\
    -lopencv_shape\
    -lopencv_stitching\
    -lopencv_video\
    -lopencv_videostab\


LIBS += -L/usr/local/lib/x86_64-linux-gnu\
    -ldlib
LIBS +=-L/home/zzy/tbb/build/linux_intel64_gcc_cc4.8_libc2.19_kernel4.2.0_release\
    -ltbb
LIBS += -L//home/zzy/QT_app/joint_yawn_speak/yawn_speak_debug\
    -lyawn_speak
LIBS += -L/home/zzy/QT_app/build-DataM_vless-Desktop_Qt_5_3_0_GCC_64bit-Debug\
    -lDataM_vless

