#ifndef INPUTPROVIDER_HPP
#define INPUTPROVIDER_HPP

#include <eigen3/Eigen/Dense>
#include <vector>
#include <memory>
#include <array>

#include "headers/inputOutput/mnist_reader.h"
#include "headers/inputOutput/cifar10_reader.hpp"

class InputProvider
{
    public:
        /// Un alias pour désigner un pointeur sur in InputProvider
        using Ptr = std::unique_ptr<InputProvider>;
        /// Un alias pour désigner un donnée (Entrée, Sortie)
        using Sample = std::pair<Eigen::MatrixXf, Eigen::MatrixXf>;
        /// Un alias pour désigner un batch de données (Entrée, Sortie)
        using Batch = std::vector<Sample>;
        /// Un alias pour désigner un minibatch de données (Entrée, Sortie)
        using Minibatch = Batch;

    public:
        InputProvider(unsigned int labelTrainSize, unsigned int labelTestSize)
            : mLabelTrainSize(labelTrainSize)
            , mLabelTestSize(labelTestSize) {}

        virtual Batch trainingBatch() const =0;
        virtual Batch testingBatch() const =0;

    protected:
        unsigned int mLabelTrainSize;
        unsigned int mLabelTestSize;

};

class MnistProvider : public InputProvider
{
    public:
        MnistProvider(std::vector<unsigned int> labels, unsigned int labelTrainSize = 60000, unsigned int labelTestSize = 10000);

        Batch trainingBatch() const;
        Batch testingBatch() const;

    private:
        std::array<unsigned int, 10>    mLabels;

        std::vector<Eigen::MatrixXf>    mImageTrain;
        Eigen::MatrixXi                 mLabelTrain;

        std::vector<Eigen::MatrixXf>    mImageTest;
        Eigen::MatrixXi                 mLabelTest;

};


#endif // INPUTPROVIDER_HPP
