#include "headers/inputprovider.hpp"


#include <iostream>

MnistProvider::MnistProvider(unsigned int labelTrainSize, unsigned int labelTestSize)
: InputProvider(labelTrainSize, labelTestSize)
{
        std::cout << "Chargement de MNIST" << std::endl;

        mnist_reader readerTrain("MNIST/train-images-60k", "MNIST/train-labels-60k");
        readerTrain.ReadMNIST(mImageTrain, mLabelTrain);

        mnist_reader readerTest("MNIST/test-images-10k", "MNIST/test-labels-10k");
        readerTest.ReadMNIST(mImageTest, mLabelTest);
}

InputProvider::Batch MnistProvider::trainingBatch() const
{
    std::cout << "Création du Batch d'entrainement du discriminateur" << std::endl;

    Batch trainingBatch;

    Eigen::MatrixXf outputTrain = Eigen::MatrixXf::Zero(1,1);
    outputTrain(0,0) = 1.0;

    for(unsigned int i(0); i < mLabelTrainSize; i++)
    {
        trainingBatch.push_back(Sample(mImageTrain[i], outputTrain));
    }

    std::cout << "Chargement du Batch d'entrainement effectué !" << std::endl;

    return trainingBatch;
}

InputProvider::Batch MnistProvider::testingBatch() const
{
    std::cout << "Création du Batch de test du discriminateur" << std::endl;

    Batch testingBatch;

    Eigen::MatrixXf outputTest = Eigen::MatrixXf::Zero(1,1);
    outputTest(0,0) = 1.0;

    for(unsigned int i(0); i < mLabelTestSize; i++)
    {
        testingBatch.push_back(Sample(mImageTest[i], outputTest));
    }

    std::cout << "Chargement du Batch de test effectué !" << std::endl;

    return testingBatch;
}

