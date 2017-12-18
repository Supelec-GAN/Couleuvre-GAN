#include "headers/neuronlayer.hpp"

//*************CONSTRUCTEUR*************
//**************************************

NeuronLayer::NeuronLayer(unsigned int inputSize, unsigned int outputSize, std::function<float(float)> activationF)
: mPoids(Eigen::MatrixXf::Random(inputSize,outputSize))
, mBiais(Eigen::MatrixXf::Random(1, outputSize))
, mActivationFun(activationF)
, mBufferActivationLevel(Eigen::MatrixXf::Zero(1, outputSize))
, mBufferInput(Eigen::MatrixXf::Zero(1, inputSize))
{}


//*************PROPAGATION**************
//**************************************

Eigen::MatrixXf NeuronLayer::process(Eigen::MatrixXf inputs)
{
    mBufferInput = inputs;
    mBufferActivationLevel = inputs*mPoids - mBiais;
    Eigen::MatrixXf output = mBufferActivationLevel;

    for(unsigned int i(0); i < output.size(); i++)
        output(0,i) = mActivationFun(output(0,i));

    return output;
}

//***********RETROPROPAGATION***********
//**************************************


Eigen::MatrixXf NeuronLayer::backProp(Eigen::MatrixXf xnPartialDerivative, float step)
{
    // Calcul de ynPartialDerivative
    Eigen::MatrixXf ynPartialDerivative = xnPartialDerivative*fnDerivativeMatrix();

    //Mise à jour des poids
    Eigen::MatrixXf wnPartialDerivative = (mBufferInput.transpose())*ynPartialDerivative;
    mPoids -= step*wnPartialDerivative;

    // Mise à jour des biais
    mBiais += step*ynPartialDerivative;

    //Retour de x(n-1)PartialDerivative
    return ynPartialDerivative*mPoids.transpose();
}

Eigen::MatrixXf NeuronLayer::fnDerivativeMatrix() const
{
    auto fnDerivated = [this] (float x, float dx)
                        {
                            return (mActivationFun(x+dx) - mActivationFun(x))/dx;
                        };

    Eigen::MatrixXf fnDerivativeMat(mBufferActivationLevel.size(),1);
    for(auto i(0); i < mBufferActivationLevel.size(); ++i)
        fnDerivativeMat(i) = fnDerivated(mBufferActivationLevel(i), 0.05);

    return Eigen::MatrixXf(fnDerivativeMat.asDiagonal());
}

void NeuronLayer::reset()
{
    mPoids = Eigen::MatrixXf::Random(mPoids.rows(), mPoids.cols());
    mBiais = Eigen::MatrixXf::Random(1,mBiais.cols());
}

int NeuronLayer::getInputSize()
{
    return (mPoids.rows());
}

//*************AUXILIAIRES**************
//**************************************

std::ostream& operator<<(std::ostream& flux, NeuronLayer n)
{
    flux << n.mPoids;
    return flux;
}
