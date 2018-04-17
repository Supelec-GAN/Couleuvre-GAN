#include <eigen3/Eigen/Dense>

#include <math.h>
#include "headers/teacher.hpp"

//#pragma mark Constructeur
Teacher::Teacher()
{}

Teacher::Teacher(NeuralNetwork::Ptr generator, NeuralNetwork::Ptr discriminator, unsigned int genFun)
: mGenerator(std::move(generator))
, mDiscriminator(std::move(discriminator))
, mErrorFunDis(Functions::coutDiscr())
{
    switch (genFun)
    {
    case 0 :
        mErrorFunGen = Functions::coutGen();
        break;
    case 1 :
        mErrorFunGen = Functions::genMinMax();
        break;
    case 2 :
        mErrorFunGen = Functions::genKLDiv();
        break;
    default :
        mErrorFunGen = Functions::coutGen();
        break;
    }
}

Teacher::Teacher(NeuralNetwork* generator, NeuralNetwork* discriminator, unsigned int genFun)
: mGenerator(generator)
, mDiscriminator(discriminator)
, mErrorFunDis(Functions::coutDiscr())
{
    switch (genFun)
    {
    case 0 :
        mErrorFunGen = Functions::coutGen();
        break;
    case 1 :
        mErrorFunGen = Functions::genMinMax();
        break;
    case 2 :
        mErrorFunGen = Functions::genKLDiv();
        break;
    default :
        mErrorFunGen = Functions::coutGen();
        break;
    }
}

//#pragma mark - Backpropagation

//#pragma mark Backprop
void Teacher::backpropDiscriminator(Eigen::MatrixXf input, Eigen::MatrixXf desiredOutput, float step, float dx)
{
	Eigen::MatrixXf xnPartialDerivative = calculateInitialErrorVector(mDiscriminator->processNetwork(input), desiredOutput, dx);
	
	propagateError(mDiscriminator, xnPartialDerivative, step);
}

void Teacher::backpropGenerator(Eigen::MatrixXf input, Eigen::MatrixXf desiredOutput, float step, float dx)
{
    Eigen::MatrixXf xnPartialDerivative = calculateInitialErrorVectorGen(mDiscriminator->processNetwork(input), desiredOutput, dx);
	xnPartialDerivative = propagateErrorDiscriminatorInvariant(xnPartialDerivative);
	propagateError(mGenerator, xnPartialDerivative, step);
}

//#pragma mark Minibatch

void Teacher::minibatchDiscriminatorBackprop(NeuralNetwork::Ptr network, Eigen::MatrixXf input,Eigen::MatrixXf desiredOutput, float step, float dx)
//Same as backpropDiscriminator but no weight updating
{
	Eigen::MatrixXf xnPartialDerivative = calculateInitialErrorVector(mDiscriminator->processNetwork(input), desiredOutput, dx);

	propagateErrorMinibatch(network, xnPartialDerivative, step);
}

void Teacher::minibatchGeneratorBackprop(NeuralNetwork::Ptr network, Eigen::MatrixXf input,Eigen::MatrixXf desiredOutput, float step, float dx)
//Same as backpropGenerator but no weight updating
{
    Eigen::MatrixXf xnPartialDerivative = calculateInitialErrorVectorGen(mDiscriminator->processNetwork(input), desiredOutput, dx);
	xnPartialDerivative = propagateErrorMinibatch(mDiscriminator, xnPartialDerivative, 0);
	propagateErrorMinibatch(network, xnPartialDerivative, step);
	
}

void Teacher::updateNetworkWeights(NeuralNetwork::Ptr network, unsigned int minibatchSize)
{
	for(auto itr = network->rbegin(); itr != network->rend(); ++itr)
		itr->updateLayerWeights(minibatchSize);
}


//#pragma mark Error Propagation

void Teacher::propagateError(NeuralNetwork::Ptr network, Eigen::MatrixXf xnPartialDerivative, float step)
{
    for(auto itr = network->rbegin(); itr != network->rend(); ++itr)
    {
		xnPartialDerivative = itr->layerBackprop(xnPartialDerivative, step);
	}
}

Eigen::MatrixXf Teacher::propagateErrorMinibatch(NeuralNetwork::Ptr network, Eigen::MatrixXf xnPartialDerivative, float step)
{
	for(auto itr = network->rbegin(); itr != network->rend(); ++itr)
	{
		xnPartialDerivative = itr->minibatchLayerBackprop(xnPartialDerivative, step);
	}
	return xnPartialDerivative;
}

//[[deprecated]] //use propagateErrorMinibatch(mDiscriminator, xnPartialDerivative, 0) instead
Eigen::MatrixXf Teacher::propagateErrorDiscriminatorInvariant(Eigen::MatrixXf xnPartialDerivative)
{
    for(auto itr = mDiscriminator->rbegin(); itr != mDiscriminator->rend(); ++itr)
    {
		xnPartialDerivative = itr->layerBackpropInvariant(xnPartialDerivative);
    }
    return xnPartialDerivative;
}


//#pragma mark initial vector calculation
//Quasiment la meme que pour le Discriminateur, à la fonction d'erreur pret
Eigen::MatrixXf Teacher::calculateInitialErrorVectorGen(Eigen::MatrixXf output, Eigen::MatrixXf desiredOutput, float dx)
{
    Eigen::MatrixXf errorVect = Eigen::MatrixXf::Zero(1, output.size());

    for(unsigned int i(0); i < output.size(); ++i)
    {
        Eigen::MatrixXf deltaX(Eigen::MatrixXf::Zero(1, output.size()));
        deltaX(i) = dx;
        errorVect(i) = std::min(100.f,(mErrorFunGen(output + deltaX, desiredOutput) - mErrorFunGen(output, desiredOutput))/dx);
    }
    return errorVect;
}

Eigen::MatrixXf Teacher::calculateInitialErrorVector(Eigen::MatrixXf output, Eigen::MatrixXf desiredOutput, float dx)
{
    Eigen::MatrixXf errorVect = Eigen::MatrixXf::Zero(1, output.size());

    for(unsigned int i(0); i < output.size(); ++i)
    {
        Eigen::MatrixXf deltaX(Eigen::MatrixXf::Zero(1, output.size()));
        deltaX(i) = dx;
        errorVect(i) = std::min(100.f,(mErrorFunDis(output + deltaX, desiredOutput) - mErrorFunDis(output, desiredOutput))/dx);
    }
    return errorVect;
}

