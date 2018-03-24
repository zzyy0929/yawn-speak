#-------------------------------------------------
#
# Project created by QtCreator 2018-01-29T15:46:26
#
#-------------------------------------------------

QT       -= gui

TARGET = yawn_speak
TEMPLATE = lib

DEFINES += YAWN_SPEAK_LIBRARY
DEFINES += QT_DEPRECATED_WARNINGS
CONFIG   += c++11
SOURCES += yawn_speak.cpp
INCLUDEPATH += /home/zzy/QT_app/Data_struct
INCLUDEPATH += /home/zzy/QT_app/DataM_vless
INCLUDEPATH += /usr/local/include/opencv
INCLUDEPATH += /usr/local/include/opencv2

LIBS += -L/home/zzy/QT_app/build-DataM_vless-Desktop_Qt_5_3_0_GCC_64bit-Debug\
    -lDataM_vless

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

HEADERS += yawn_speak.h\
        yawn_speak_global.h

unix {
    target.path = /usr/lib
    INSTALLS += target
}
