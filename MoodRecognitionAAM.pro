#-------------------------------------------------
#
# Project created by QtCreator 2014-10-25T14:44:37
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = MoodRecognitionAAM
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    aam.cpp \
    imagepreprocessing.cpp \
    comparepoint.cpp \
    classifier.cpp

HEADERS  += mainwindow.h \
    aam.h \
    imagepreprocessing.h \
    comparepoint.h \
    classifier.h

FORMS    += mainwindow.ui

INCLUDEPATH += /usr/local/include/opencv2
LIBS += -L/usr/local/lib
LIBS += -lopencv_core
LIBS += -lopencv_imgproc
LIBS += -lopencv_highgui
LIBS += -lopencv_ml
LIBS += -lopencv_video
LIBS += -lopencv_features2d
LIBS += -lopencv_calib3d
LIBS += -lopencv_objdetect
LIBS += -lopencv_flann
LIBS += -lopencv_imgcodecs
LIBS += -lopencv_photo
LIBS += -lopencv_shape
LIBS += -lopencv_stitching
LIBS += -lopencv_superres
LIBS += -lopencv_ts
LIBS += -lopencv_video
LIBS += -lopencv_videoio
LIBS += -lopencv_videostab
