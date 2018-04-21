TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle

QT += core
QT += gui
QT += widgets

QMAKE_CXXFLAGS += -fopenmp
LIBS += -fopenmp

HEADERS += \
    headers/inputOutput/CSVFile.h \
    headers/functions.hpp \
    headers/inputOutput/mnist_reader.h \
    headers/neuralnetwork.hpp \
    headers/neuronlayer.hpp \
    headers/teacher.hpp \
    headers/application.hpp \
    headers/inputOutput/errorcollector.hpp \
    headers/functions.hpp \
    headers/inputOutput/statscollector.hpp \
    headers/inputOutput/inputprovider.hpp \
    headers/inputOutput/cifar10_reader.hpp \
    headers/inputOutput/cifar10provider.hpp \
    headers/convolution.hpp \
    headers/layers/convolutionallayer.hpp \
    headers/layers/fullconnectedlayer.hpp \
    headers/layers/noisylayer.h \
    headers/layers/maxpoolinglayer.hpp \
    headers/layers/zeropadlayer.hpp

SOURCES += \
    sources/functions.cpp \
    sources/main.cpp \
    sources/inputOutput/mnist_reader.cpp \
    sources/neuralnetwork.cpp \
    sources/neuronlayer.cpp \
    sources/teacher.cpp \
    headers/neuralnetwork.inl \
    sources/inputOutput/errorcollector.cpp \
    sources/inputOutput/statscollector.cpp \
    sources/application.cpp \
    sources/inputOutput/inputprovider.cpp \
    sources/inputOutput/cifar10provider.cpp \
    sources/convolution.cpp \
    sources/layers/convolutionallayer.cpp \
    sources/layers/fullconnectedlayer.cpp \
    sources/layers/noisylayer.cpp \
    sources/layers/maxpoolinglayer.cpp \
    sources/layers/zeropadlayer.cpp

DISTFILES += \
    MNIST/test-images-10k \
    MNIST/test-labels-10k \
    MNIST/train-images-60k \
    MNIST/train-labels-60k
