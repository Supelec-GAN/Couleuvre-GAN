#include <eigen3/Eigen/Dense>

#include "headers/teacher.hpp"

//#pragma mark Constructeur
Teacher::Teacher()
{}

Teacher::Teacher(NeuralNetwork::Ptr generator, NeuralNetwork::Ptr discriminator)
: mGenerator(std::move(generator))
, mDiscriminator(std::move(discriminator))
, mErrorFun(Functions::l2Norm())
{}

Teacher::Teacher(NeuralNetwork* generator, NeuralNetwork* discriminator)
: mGenerator(generator)
, mDiscriminator(discriminator)
, mErrorFun(Functions::coutDiscr())
{}

//#pragma mark - Backpropagation

//#pragma mark Backprop
void Teacher::backpropDiscriminator(Eigen::MatrixXf input, Eigen::MatrixXf desiredOutput, float step, float dx)
{
	Eigen::MatrixXf xnPartialDerivative = calculateInitialErrorVector(mDiscriminator->processNetwork(input), desiredOutput, dx);
	
	propagateError(mDiscriminator, xnPartialDerivative, step);
}

void Teacher::backpropGenerator(Eigen::MatrixXf input, Eigen::MatrixXf desiredOutput, float step, float dx)
{
	Eigen::MatrixXf xnPartialDerivative = calculateInitialErrorVector(mDiscriminator->processNetwork(input), desiredOutput, dx);
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
	Eigen::MatrixXf xnPartialDerivative = calculateInitialErrorVector(mDiscriminator->processNetwork(input), desiredOutput, dx);
	xnPartialDerivative = propagateErrorMinibatch(mDiscriminator, xnPartialDerivative, 0);
	propagateErrorMinibatch(network, xnPartialDerivative, step);
	
}

void Teacher::updateNetworkWeights(NeuralNetwork::Ptr network)
{
	for(auto itr = network->rbegin(); itr != network->rend(); ++itr)
		itr->updateLayerWeights();
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

[[deprecated]] //use propagateErrorMinibatch(mDiscriminator, xnPartialDerivative, 0) instead
Eigen::MatrixXf Teacher::propagateErrorDiscriminatorInvariant(Eigen::MatrixXf xnPartialDerivative)
{
    for(auto itr = mDiscriminator->rbegin(); itr != mDiscriminator->rend(); ++itr)
    {
		xnPartialDerivative = itr->layerBackpropInvariant(xnPartialDerivative);
    }
    return xnPartialDerivative;
}


//#pragma mark initial vector calculation

Eigen::MatrixXf Teacher::calculateInitialErrorVectorGen(Eigen::MatrixXf output, Eigen::MatrixXf desiredOutput, float dx)
{
    Eigen::MatrixXf errorVect = Eigen::MatrixXf::Zero(1, output.size());
    Eigen::MatrixXf discrOutput = mDiscriminator->processNetwork(output);

    for(unsigned int i(0); i < output.size(); ++i)
    {
        Eigen::MatrixXf deltaX(Eigen::MatrixXf::Zero(1, output.size()));
        deltaX(i) = dx;
        errorVect(i) = (mErrorFun(mDiscriminator->processNetwork(output + deltaX), desiredOutput) - mErrorFun(discrOutput, desiredOutput))/dx;
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
        errorVect(i) = (mErrorFun(output + deltaX, desiredOutput) - mErrorFun(output, desiredOutput))/dx;
    }
    return errorVect;
}

