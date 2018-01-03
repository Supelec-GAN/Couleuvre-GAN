#include <eigen3/Eigen/Dense>

#include "headers/teacher.hpp"


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

#pragma mark - Backpropagation

#pragma mark Generator

void Teacher::backpropGenerator(Eigen::MatrixXf input, Eigen::MatrixXf desiredOutput, float step, float dx)
{
    Eigen::MatrixXf xnPartialDerivative = calculateInitialErrorVector(mDiscriminator->processNetwork(input), desiredOutput, dx);
    xnPartialDerivative = propagateErrorDiscriminatorInvariant(xnPartialDerivative);
    propagateErrorGenerator(xnPartialDerivative, step);
}

void Teacher::propagateErrorGenerator(Eigen::MatrixXf xnPartialDerivative, float step)
{
    for(auto itr = mGenerator->rbegin(); itr != mGenerator->rend(); ++itr)
    {
		xnPartialDerivative = itr->layerBackprop(xnPartialDerivative, step);
	}
}

#pragma mark Discriminator

void Teacher::backpropDiscriminator(Eigen::MatrixXf input, Eigen::MatrixXf desiredOutput, float step, float dx)
{
    Eigen::MatrixXf xnPartialDerivative = calculateInitialErrorVector(mDiscriminator->processNetwork(input), desiredOutput, dx);

    propagateErrorDiscriminator(xnPartialDerivative, step);
}

void Teacher::propagateErrorDiscriminator(Eigen::MatrixXf xnPartialDerivative, float step)
{
    for(auto itr = mDiscriminator->rbegin(); itr != mDiscriminator->rend(); ++itr)
    {
        xnPartialDerivative = itr->layerBackprop(xnPartialDerivative, step);
	}
}

Eigen::MatrixXf Teacher::propagateErrorDiscriminatorInvariant(Eigen::MatrixXf xnPartialDerivative)
{
    for(auto itr = mDiscriminator->rbegin(); itr != mDiscriminator->rend(); ++itr)
    {
        xnPartialDerivative = itr->layerBackpropInvariant(xnPartialDerivative);
    }
    return xnPartialDerivative;
}

#pragma mark Minibatch

void Teacher::miniBatchBackProp(Eigen::VectorXf input,Eigen::VectorXf desiredOutput, float step, float dx)
{
#warning Japillow must implement
	throw std::logic_error("Not implemented yet");
//	Eigen::VectorXf xnPartialDerivative = errorVector(mNetwork->process(input), desiredOutput, dx);
//	for(auto itr = mNetwork->rbegin(); itr != mNetwork->rend(); ++itr)
//	{
//		xnPartialDerivative = itr->layerBackProp(xnPartialDerivative, step);
//	}
}

void Teacher::updateNetworkWeights()
{
#warning Japillow must implement
	throw std::logic_error("Not implemented yet");
//	for(auto itr = mNetwork->rbegin(); itr != mNetwork->rend(); ++itr)
//		itr->updateLayerWeights();
}

#pragma mark initial vector calculation

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
