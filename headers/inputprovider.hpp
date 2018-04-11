#ifndef INPUTPROVIDER_HPP
#define INPUTPROVIDER_HPP

#include <eigen3/Eigen/Dense>
#include <vector>

class InputProvider
{
    public:
        /// Un alias pour désigner un donnée (Entrée, Sortie)
        using Sample = std::pair<Eigen::MatrixXf, Eigen::MatrixXf>;
        /// Un alias pour désigner un batch de données (Entrée, Sortie)
        using Batch = std::vector<Sample>;
        /// Un alias pour désigner un minibatch de données (Entrée, Sortie)
        using Minibatch = Batch;

    public:
        virtual Batch trainingBatch() const =0;
        virtual Batch testingBatch() const =0;

};

class MnistProvider : public InputProvider
{
    public:
        MnistProvider();

        Batch trainingBatch(unsigned int labelTrainSize = 60000) const;
        Batch testingBatch(unsigned int labelTestSize = 10000) const;

    private:
        std::vector<Eigen::MatrixXf>    mImageTrain;
        Eigen::MatrixXi                 mLabelTrain;

        std::vector<Eigen::MatrixXf>    mImageTest;
        Eigen::MatrixXi                 mLabelTest;

};

#endif // INPUTPROVIDER_HPP
