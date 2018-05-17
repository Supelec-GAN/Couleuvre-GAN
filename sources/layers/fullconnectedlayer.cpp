#include "headers/layers/fullconnectedlayer.hpp"
#include <cmath>

//*************CONSTRUCTEUR*************
//**************************************

FullConnectedLayer::FullConnectedLayer(unsigned int inputSize, unsigned int outputSize, std::function<float(float)> activationF, unsigned int descentType)
: mWeight(Eigen::MatrixXf::Random(inputSize,outputSize))
, mBias(Eigen::MatrixXf::Random(1, outputSize)) 				//ligne
, mActivationFun(activationF)
, mBufferActivationLevel(Eigen::MatrixXf::Zero(1, outputSize))	//ligne
, mBufferInput(Eigen::MatrixXf::Zero(1, inputSize))				//ligne
, mSumWeightVariation(Eigen::MatrixXf::Zero(inputSize, outputSize))
, mSumBiasVariation(Eigen::MatrixXf::Zero(1, outputSize))
, mAdaptativeWeightStep(Eigen::MatrixXf::Constant(inputSize, outputSize, 1))
, mAdaptativeBiasStep(Eigen::MatrixXf::Constant(1, outputSize, 1))
, mAdaptativeWeightSecondStep(Eigen::MatrixXf::Constant(inputSize, outputSize, 1))
, mAdaptativeBiasSecondStep(Eigen::MatrixXf::Constant(1, outputSize, 1))
, mDescentType(descentType)
, mUpdateCount(0)
{}

FullConnectedLayer::FullConnectedLayer(unsigned int inputSize, unsigned int outputSize, Eigen::MatrixXf weight, Eigen::MatrixXf bias, std::function<float(float)> activationF, unsigned int descentType)
: mWeight(weight)
, mBias(bias) 				//ligne
, mActivationFun(activationF)
, mBufferActivationLevel(Eigen::MatrixXf::Zero(1, outputSize))	//ligne
, mBufferInput(Eigen::MatrixXf::Zero(1, inputSize))				//ligne
, mSumWeightVariation(Eigen::MatrixXf::Zero(inputSize, outputSize))
, mSumBiasVariation(Eigen::MatrixXf::Zero(1, outputSize))
, mAdaptativeWeightStep(Eigen::MatrixXf::Constant(inputSize, outputSize, 1))
, mAdaptativeBiasStep(Eigen::MatrixXf::Constant(1, outputSize, 1))
, mAdaptativeWeightSecondStep(Eigen::MatrixXf::Constant(inputSize, outputSize, 1))
, mAdaptativeBiasSecondStep(Eigen::MatrixXf::Constant(1, outputSize, 1))
, mDescentType(descentType)
, mUpdateCount(0)
{}

FullConnectedLayer::~FullConnectedLayer(){}

//*************PROPAGATION**************
//**************************************

Eigen::MatrixXf FullConnectedLayer::processLayer(Eigen::MatrixXf inputs)
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


Eigen::MatrixXf FullConnectedLayer::layerBackprop(Eigen::MatrixXf xnPartialDerivative, float step)
{
    // Calcul de ynPartialDerivative
    Eigen::MatrixXf ynPartialDerivative = xnPartialDerivative*fnDerivativeMatrix();

    //Mise à jour des poids
    Eigen::MatrixXf wnPartialDerivative = (mBufferInput.transpose())*ynPartialDerivative;
    switch(mDescentType)
    {
    case 1:     // RMS Prop
        updateFirstMomentStep(wnPartialDerivative, ynPartialDerivative, step);
        mSumBiasVariation += ((1.0/(sqrt(mAdaptativeBiasStep.array()+0.000001)))*ynPartialDerivative.array()).matrix();
        mSumWeightVariation += ((1.0/(sqrt(mAdaptativeWeightStep.array()+0.000001)))*wnPartialDerivative.array()).matrix();
        updateLayerWeights();
        break;
    case 2:     // Adam
        mUpdateCount++;
        updateFirstMomentStep(wnPartialDerivative, ynPartialDerivative, step);
        updateSecondMomentStep(wnPartialDerivative, ynPartialDerivative, step);
        mSumBiasVariation += ((1.0/(sqrt(((mAdaptativeBiasSecondStep).array())+0.000001)))*mAdaptativeBiasStep.array()).matrix();
        mSumWeightVariation += ((1.0/(sqrt(((mAdaptativeWeightSecondStep).array())+0.000001)))*mAdaptativeWeightStep.array()).matrix();
        updateLayerWeights();
        break;
    default:
        mSumBiasVariation += step*ynPartialDerivative;
        mSumWeightVariation += step*wnPartialDerivative;
        updateLayerWeights();
        break;
    }
    //Retour de x(n-1)PartialDerivative
    return ynPartialDerivative*mWeight.transpose();
}

[[deprecated]] //use minibatchLayerBackprop instead
Eigen::MatrixXf FullConnectedLayer::layerBackpropInvariant(Eigen::MatrixXf xnPartialDerivative)
{
    // Calcul de ynPartialDerivative
    Eigen::MatrixXf ynPartialDerivative = xnPartialDerivative*fnDerivativeMatrix();

    //Retour de x(n-1)PartialDerivative
    return ynPartialDerivative*mWeight.transpose();
}

Eigen::MatrixXf FullConnectedLayer::minibatchLayerBackprop(Eigen::MatrixXf xnPartialDerivative, float step)
{
    //Same as layerBackprop but no weight updating

    // Calcul de ynPartialDerivative
    Eigen::MatrixXf ynPartialDerivative = xnPartialDerivative*fnDerivativeMatrix();

    //Calcul de wnPartialDerivative et somation des erreurs
    Eigen::MatrixXf wnPartialDerivative = (mBufferInput.transpose())*ynPartialDerivative;
    mSumBiasVariation += step*ynPartialDerivative;
    mSumWeightVariation += step*wnPartialDerivative;
    //Pas de mise à jour au sein de la backprop

    //Retour de x(n-1)PartialDerivative
    return ynPartialDerivative*mWeight.transpose();
}

//****************AUTRES****************
//**************************************

void FullConnectedLayer::updateLayerWeights(unsigned int minibatchSize)
{
    mWeight -= mSumWeightVariation/minibatchSize;
    mBias += mSumBiasVariation/minibatchSize;

    //reset des buffer
    mSumWeightVariation.setZero() ;
    mSumBiasVariation.setZero();
}

Eigen::MatrixXf FullConnectedLayer::fnDerivativeMatrix() const
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

void FullConnectedLayer::updateFirstMomentStep(Eigen::MatrixXf wnPartialDerivative, Eigen::MatrixXf ynPartialDerivative,  float step)
{
     mAdaptativeWeightStep = step*mAdaptativeWeightStep + (1-step)*abs(wnPartialDerivative.array()).matrix();
     mAdaptativeBiasStep = step*mAdaptativeBiasStep + (1-step)*abs(ynPartialDerivative.array()).matrix();
}

void FullConnectedLayer::updateSecondMomentStep(Eigen::MatrixXf wnPartialDerivative, Eigen::MatrixXf ynPartialDerivative, float step)
{
    mAdaptativeWeightSecondStep = 0.99*mAdaptativeWeightStep + (1-0.99)*(wnPartialDerivative.array()*wnPartialDerivative.array()).matrix();
    mAdaptativeBiasSecondStep = 0.99*mAdaptativeBiasStep + (1-0.99)*(ynPartialDerivative.array()*ynPartialDerivative.array()).matrix();
}

void FullConnectedLayer::reset()
{
    mWeight = Eigen::MatrixXf::Random(mWeight.rows(), mWeight.cols());
    mBias = Eigen::MatrixXf::Random(1,mBias.cols());
}

int FullConnectedLayer::getInputSize()
{
    return (static_cast<int>(mWeight.rows()));
}


/*std::ostream& operator<<(std::ostream& flux, FullConnectedLayer n)
{
    for(int i(0); i < n.mWeight.rows(); i++)
    {
        for(int j(0); j < n.mWeight.cols(); j++)
        {
            flux << n.mWeight(i,j);
            flux << "; ";
        }
        flux << "\n";
    }
    flux << "\n";
    for(int j(0); j < n.mWeight.cols(); j++)
    {
        flux << n.mBias(j);
        flux << "; ";
    }
    flux << "\n";
    return flux;
}*/
