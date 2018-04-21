#include "headers/inputOutput/inputprovider.hpp"


#include <iostream>

MnistProvider::MnistProvider(std::vector<unsigned int> labels, unsigned int labelTrainSize, unsigned int labelTestSize)
: InputProvider(labelTrainSize, labelTestSize)
, mLabels{0}
{
        std::cout << "Chargement de MNIST" << std::endl;

        mnist_reader readerTrain("MNIST/train-images-60k", "MNIST/train-labels-60k");
        readerTrain.ReadMNIST(mImageTrain, mLabelTrain);

        mnist_reader readerTest("MNIST/test-images-10k", "MNIST/test-labels-10k");
        readerTest.ReadMNIST(mImageTest, mLabelTest);

        for(size_t i(0); i< labels.size(); i++)
            mLabels[labels[i]] = 1;
}

InputProvider::Batch MnistProvider::trainingBatch() const
{
    std::cout << "Création du Batch d'entrainement du discriminateur" << std::endl;

    Batch trainingBatch;

    Eigen::MatrixXf outputTrain = Eigen::MatrixXf::Zero(1,1);
    outputTrain(0,0) = 1.0;

    unsigned int compteur(0);
    for(unsigned int i(0); i < 60000 && compteur < mLabelTrainSize; i++)
    {
        if(mLabels[mLabelTrain(i)])
        {
            trainingBatch.push_back(Sample(mImageTrain[i], outputTrain));
            compteur++;
        }
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

    unsigned int compteur(0);
    for(unsigned int i(0); i < mLabelTestSize; i++)
    {
        if(mLabels[mLabelTest(i)])
        {
           testingBatch.push_back(Sample(mImageTest[i], outputTest));
            compteur++;
        }
    }
    std::cout << "Chargement du Batch de test effectué !" << std::endl;

    return testingBatch;
}

