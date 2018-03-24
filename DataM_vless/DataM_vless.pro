#-------------------------------------------------
#
# Project created by QtCreator 2017-11-13T10:10:02
#
#-------------------------------------------------

QT       -= gui

TARGET = DataM_vless
TEMPLATE = lib

DEFINES += DATAM_VLESS_LIBRARY

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        datam_vless.cpp

HEADERS += \
        datam_vless.h \
        datam_vless_global.h 
INCLUDEPATH += /usr/local/include/opencv
INCLUDEPATH += /usr/local/include/opencv2

LIBS += -L/usr/local/lib\
    -lopencv_core\
    -lopencv_imgproc\
    -lopencv_highgui\
    -lopencv_features2d\
    -lopencv_calib3d\
    -lopencv_imgcodecs\
    -lopencv_videoio\
    -lopencv_objdetect

#unix {
#    target.path = /usr/lib
#    INSTALLS += target
#}
