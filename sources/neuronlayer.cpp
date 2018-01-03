#include "headers/neuronlayer.hpp"

//*************CONSTRUCTEUR*************
//**************************************

NeuronLayer::NeuronLayer(unsigned int inputSize, unsigned int outputSize, std::function<float(float)> activationF)
: mWeight(Eigen::MatrixXf::Random(inputSize,outputSize))
, mBias(Eigen::MatrixXf::Random(1, outputSize)) 				//ligne
, mActivationFun(activationF)
, mBufferActivationLevel(Eigen::MatrixXf::Zero(1, outputSize))	//ligne
, mBufferInput(Eigen::MatrixXf::Zero(1, inputSize))				//ligne
, mSumWeightVariation(Eigen::MatrixXf::Zero(inputSize, outputSize))
, mSumBiasVariation(Eigen::MatrixXf::Zero(1, outputSize))
{}


//*************PROPAGATION**************
//**************************************

Eigen::MatrixXf NeuronLayer::processLayer(Eigen::MatrixXf inputs)
{
    mBufferInput = inputs;
    mBufferActivationLevel = inputs*mWeight - mBias;
    Eigen::MatrixXf output = mBufferActivationLevel;

    for(unsigned int i(0); i < output.size(); i++)
        output(0,i) = mActivationFun(output(0,i));

    return output;
}

//***********RETROPROPAGATION***********
//**************************************


Eigen::MatrixXf NeuronLayer::layerBackprop(Eigen::MatrixXf xnPartialDerivative, float step)
{
    // Calcul de ynPartialDerivative
    Eigen::MatrixXf ynPartialDerivative = xnPartialDerivative*fnDerivativeMatrix();

    //Mise à jour des poids
    Eigen::MatrixXf wnPartialDerivative = (mBufferInput.transpose())*ynPartialDerivative;
    mWeight -= step*wnPartialDerivative;

    // Mise à jour des biais
    mBias += step*ynPartialDerivative;

    //Retour de x(n-1)PartialDerivative
    return ynPartialDerivative*mWeight.transpose();
}

Eigen::MatrixXf NeuronLayer::layerBackpropInvariant(Eigen::MatrixXf xnPartialDerivative)
{
    // Calcul de ynPartialDerivative
    Eigen::MatrixXf ynPartialDerivative = xnPartialDerivative*fnDerivativeMatrix();

    //Retour de x(n-1)PartialDerivative
    return ynPartialDerivative*mWeight.transpose();
}

//****************AUTRES****************
//**************************************

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
    mWeight = Eigen::MatrixXf::Random(mWeight.rows(), mWeight.cols());
    mBias = Eigen::MatrixXf::Random(1,mBias.cols());
}

int NeuronLayer::getInputSize()
{
    return (static_cast<int>(mWeight.rows()));
}

//*************AUXILIAIRES**************
//**************************************

std::ostream& operator<<(std::ostream& flux, NeuronLayer n)
{
    flux << n.mWeight;
    return flux;
}
