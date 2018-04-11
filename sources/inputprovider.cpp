#include "headers/inputprovider.hpp"

#include "headers/mnist_reader.h"

MnistProvider::MnistProvider()
{
        mnist_reader readerTrain("MNIST/train-images-60k", "MNIST/train-labels-60k");
        readerTrain.ReadMNIST(mImageTrain, mLabelTrain);

        mnist_reader readerTest("MNIST/test-images-10k", "MNIST/test-labels-10k");
        readerTest.ReadMNIST(mImageTest, mLabelTest);
}

InputProvider::Batch MnistProvider::trainingBatch(unsigned int labelTrainSize) const
{
    Batch trainingBatch;

    Eigen::MatrixXf outputTrain = Eigen::MatrixXf::Zero(1,1);
    outputTrain(0,0) = 1.0;

    for(unsigned int i(0); i < labelTrainSize; i++)
    {
        trainingBatch.push_back(Sample(mImageTrain[i], outputTrain));
    }

    return trainingBatch;
}

InputProvider::Batch MnistProvider::testingBatch(unsigned int labelTestSize) const
{
    Batch testingBatch;

    Eigen::MatrixXf outputTest = Eigen::MatrixXf::Zero(1,1);
    outputTest(0,0) = 1.0;

    for(unsigned int i(0); i < labelTestSize; i++)
    {
        testingBatch.push_back(Sample(mImageTest[i], outputTest));
    }

    return testingBatch;
}
