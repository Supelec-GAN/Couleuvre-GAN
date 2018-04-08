TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle

QT += core
QT += gui
QT += widgets

QMAKE_CXXFLAGS += -fopenmp -Ofast
LIBS += -fopenmp

HEADERS += \
    headers/CSVFile.h \
    headers/functions.hpp \
    headers/mnist_reader.h \
    headers/neuralnetwork.hpp \
    headers/neuronlayer.hpp \
    headers/teacher.hpp \
    headers/application.hpp \
    headers/errorcollector.hpp \
    headers/functions.hpp \
    headers/statscollector.hpp \
    headers/convolution.hpp \
    headers/convolutionallayer.hpp \
    headers/fullconnectedlayer.hpp

SOURCES += \
    sources/functions.cpp \
    sources/main.cpp \
    sources/mnist_reader.cpp \
    sources/neuralnetwork.cpp \
    sources/neuronlayer.cpp \
    sources/teacher.cpp \
    headers/neuralnetwork.inl \
    sources/errorcollector.cpp \
    sources/statscollector.cpp \
    sources/application.cpp \
    sources/convolution.cpp \
    sources/convolutionallayer.cpp \
    sources/fullconnectedlayer.cpp

INCLUDEPATH += D:\Users\Kvykvzx\Projet\C++\SupelecGAN

DISTFILES += \
    MNIST/test-images-10k \
    MNIST/test-labels-10k \
    MNIST/train-images-60k \
    MNIST/train-labels-60k
