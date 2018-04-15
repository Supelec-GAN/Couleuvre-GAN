TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle

QT += core
QT += gui
QT += widgets

QMAKE_CXXFLAGS += -fopenmp
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
    headers/inputprovider.hpp \
    headers/cifar10_reader.hpp \
    headers/cifar10provider.hpp

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
    sources/inputprovider.cpp \
    sources/cifar10provider.cpp

DISTFILES += \
    MNIST/test-images-10k \
    MNIST/test-labels-10k \
    MNIST/train-images-60k \
    MNIST/train-labels-60k
