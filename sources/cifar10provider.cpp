#include "headers/cifar10provider.hpp"

Cifar10Provider::Cifar10Provider(CifarLabel labels, unsigned int labelTrainSize, unsigned int labelTestSize)
: InputProvider(labelTrainSize, labelTestSize)
, mDataset(cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>())
, mLabels(labels)
{
    if(mLabelTrainSize > 60000)
        throw std::logic_error("Erreur : dépassement d'indice sur le batch de train");
    if(mLabelTestSize > 10000)
        throw std::logic_error("Erreur : dépassement d'indice sur le batch de test");
}

InputProvider::Batch Cifar10Provider::trainingBatch() const
{
    std::cout << "Création du Batch d'entrainement Cifar10 du discriminateur" << std::endl;

    Batch trainingBatch;

    Eigen::MatrixXf outputTrain = Eigen::MatrixXf::Zero(1,1);
    outputTrain(0,0) = 1.0;

    for(unsigned int i(0); i < mLabelTrainSize; i++)
    {
        std::cout << "Chargement de l'image no " << i << std::endl;
        trainingBatch.push_back(Sample(getMatrix(i, true), outputTrain));
    }

    std::cout << "Chargement du Batch d'entrainement Cifar10 effectué !" << std::endl;

    return trainingBatch;
}

InputProvider::Batch Cifar10Provider::testingBatch() const
{
    std::cout << "Création du Batch de test Cifar10 du discriminateur" << std::endl;

    Batch testBatch;

    Eigen::MatrixXf outputTest = Eigen::MatrixXf::Zero(1,1);
    outputTest(0,0) = 1.0;

    for(unsigned int i(0); i < mLabelTrainSize; i++)
    {
        testBatch.push_back(Sample(getMatrix(i, false), outputTest));
    }

    std::cout << "Chargement du Batch de test Cifar10 effectué !" << std::endl;

    return testBatch;
}

Eigen::MatrixXf Cifar10Provider::getMatrix(unsigned int index, bool isTrainOrTestRequired) const
{
    // Le 3072 est hardcodé car c'est la taille d'une image cifar (32 * 32 = 1024 (nombre de pixels) et 1024 * 3 = 3072 (3 couleurs))
    unsigned int cifarSize = 3072;

    auto dSet = isTrainOrTestRequired ? &mDataset.training_images : &mDataset.test_images;

    Eigen::MatrixXf mat = Eigen::MatrixXf::Zero(1,cifarSize);
    for(unsigned int pixel(0); pixel < cifarSize; pixel++)
    {
        mat(0,pixel) = (*dSet)[index][pixel];
    }

    return mat;
}

bool Cifar10Provider::matchLabelWithId(CifarLabel label, uint8_t id)
{
    // Grâce à l'operation &, on regarde si id correspond à un flag actif (i.e bit à 1) dans label
    auto match = (label & static_cast<Cifar10Provider::CifarLabel>(1 << id));

    // On convertit le résultat en booléen : on return donc false quand tous les bits sont à 0 (i.e aucun flag correspondant dans label)
    return static_cast<bool>(match);
}

std::ostream& operator<<(std::ostream& flux, Cifar10Provider::CifarLabel label)
{
    flux << static_cast<int>(label);
    return flux;
}
