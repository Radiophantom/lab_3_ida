QT -= gui
QT += widgets
QT += core
#TEMPLATE = app

CONFIG += c++11 console
CONFIG -= app_bundle

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        main.cpp

INCLUDEPATH +="/usr/local/include/opencv4"
LIBS += -L"/usr/local/lib/"
LIBS += \
   -lopencv_core \
   -lopencv_highgui \
   -lopencv_imgproc \
   -lopencv_imgcodecs \
   -lopencv_calib3d \
   -lopencv_features2d \

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
