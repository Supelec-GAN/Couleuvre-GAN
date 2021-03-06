#include "headers/inputOutput/cifar10provider.hpp"

Cifar10Provider::Cifar10Provider(CifarLabel labels, unsigned int labelTrainSize, unsigned int labelTestSize)
: InputProvider(labelTrainSize, labelTestSize)
, mDataset(cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>())
, mLabels(labels)
{
    if(mLabelTrainSize > 50000)
        throw std::logic_error("Erreur : dépassement d'indice sur le batch de train");
    if(mLabelTestSize > 10000)
        throw std::logic_error("Erreur : dépassement d'indice sur le batch de test");
}

InputProvider::Batch Cifar10Provider::trainingBatch(bool greyLevel) const
{
    std::cout << "Création du Batch d'entrainement Cifar10 du discriminateur" << std::endl;

    Batch trainingBatch;

    Eigen::MatrixXf outputTrain = Eigen::MatrixXf::Zero(1,1);
    outputTrain(0,0) = 1.0;

    // Compteur permet de compter le nombre d'images dans le batch, pour ne pas dépasser mLabelTrainSize
    unsigned int compteur(0);
    for(unsigned int i(0); i < 50000 && compteur < mLabelTrainSize; i++)
    {
        if(Cifar10Provider::matchLabelWithId(mLabels, mDataset.training_labels[i]))
        {
            trainingBatch.push_back(Sample(getMatrix(i, true, greyLevel), outputTrain));
            compteur++;
        }
    }

    std::cout << "Chargement du Batch d'entrainement Cifar10 effectué !" << std::endl;
    return trainingBatch;
}

InputProvider::Batch Cifar10Provider::testingBatch(bool greyLevel) const
{

    std::cout << "Création du Batch de test Cifar10 du discriminateur" << std::endl;

    Batch testBatch;

    Eigen::MatrixXf outputTest = Eigen::MatrixXf::Zero(1,1);
    outputTest(0,0) = 1.0;

    // Compteur permet de compter le nombre d'images dans le batch, pour ne pas dépasser mLabelTestSize
    unsigned int compteur(0);
    for(unsigned int i(0); i < 50000 && compteur < mLabelTestSize; i++)
    {
        if(Cifar10Provider::matchLabelWithId(mLabels, mDataset.test_labels[i]))
        {
            testBatch.push_back(Sample(getMatrix(i, true, greyLevel), outputTest));
            compteur++;
        }
    }

    std::cout << "Chargement du Batch de test Cifar10 effectué !" << std::endl;

    return testBatch;
}

Eigen::MatrixXf Cifar10Provider::getMatrix(unsigned int index, bool isTrainOrTestRequired, bool greyLevel) const
{
    // Le 3072 est hardcodé car c'est la taille d'une image cifar (32 * 32 = 1024 (nombre de pixels) et 1024 * 3 = 3072 (3 couleurs))
    unsigned int cifarSize;
    auto dSet = isTrainOrTestRequired ? &mDataset.training_images : &mDataset.test_images;

    if(greyLevel)
    {
        cifarSize = 1024;
        Eigen::MatrixXf mat = Eigen::MatrixXf::Zero(1,cifarSize);
        for(unsigned int pixel(0); pixel < cifarSize; pixel++)
        {
            mat(0,pixel) = (*dSet)[index][pixel]*0.2126
                         + (*dSet)[index][pixel + 1024]*0.7152
                         + (*dSet)[index][pixel + 2048]*0.0722;
        }
        return mat;
    }
    else
    {
        cifarSize = 3072;
        Eigen::MatrixXf mat = Eigen::MatrixXf::Zero(1,cifarSize);
        for(unsigned int pixel(0); pixel < cifarSize; pixel++)
        {
            mat(0,pixel) = (*dSet)[index][pixel];
        }
        return mat;
    }
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
