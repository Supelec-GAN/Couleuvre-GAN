#include "headers\noisylayer.h"

NoisyLayer::NoisyLayer(unsigned int inputSize, unsigned int outputSize, std::function<float(float)> activationF)
: FullConnectedLayer(inputSize, outputSize, activationF)
, mNoiseWeight(Eigen::MatrixXf::Random(1, outputSize))
, mBufferNoise(Eigen::MatrixXf::Zero(mNoiseWeight.rows(), 1))
, mSumNoiseVariation(Eigen::MatrixXf::Zero(1, outputSize))
{

}


Eigen::MatrixXf NoisyLayer::processLayer(Eigen::MatrixXf inputs)
{
    mBufferInput = inputs;
    mBufferNoise = Eigen::VectorXf::Random((mNoiseWeight.rows(), 1));
    mBufferActivationLevel = inputs*mWeight + mBufferNoise.asDiagonal()*mNoiseWeight - mBias;
    Eigen::MatrixXf output = mBufferActivationLevel;

    for(unsigned int i(0); i < output.size(); i++)
        output(0,i) = mActivationFun(output(0,i));

    return output;
}


Eigen::MatrixXf NoisyLayer::layerBackprop(Eigen::MatrixXf xnPartialDerivative, float step)
{
    // Calcul de ynPartialDerivative
    Eigen::MatrixXf ynPartialDerivative = xnPartialDerivative*fnDerivativeMatrix();

    //Mise à jour des poids
    Eigen::MatrixXf wnPartialDerivative = (mBufferInput.transpose())*ynPartialDerivative;
    Eigen::MatrixXf noisePartialDerivative = (mBufferNoise.transpose())*ynPartialDerivative;
    mSumBiasVariation += step*ynPartialDerivative;
    mSumWeightVariation += step*wnPartialDerivative;
    mSumNoiseVariation += step*noisePartialDerivative;
    updateLayerWeights();

    //Retour de x(n-1)PartialDerivative
    return ynPartialDerivative*mWeight.transpose();
}

void NoisyLayer::updateLayerWeights(unsigned int minibatchSize)
{
    FullConnectedLayer::updateLayerWeights(minibatchSize);

    mNoiseWeight -= mSumNoiseVariation/minibatchSize;

    mSumWeightVariation.setZero();
}
