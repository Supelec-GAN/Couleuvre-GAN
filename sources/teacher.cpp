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

void Teacher::backPropGen(Eigen::MatrixXf input, Eigen::MatrixXf desiredOutput, float step, float dx)
{
    Eigen::MatrixXf xnPartialDerivative = errorVectorGen(input, desiredOutput, dx);

    propErrorGen(xnPartialDerivative, step);
}


void Teacher::propErrorGen(Eigen::MatrixXf xnPartialDerivative, float step)
{
    for(auto itr = mGenerator->rbegin(); itr != mGenerator->rend(); ++itr)
    {
        xnPartialDerivative = itr->backProp(xnPartialDerivative, step);
    }
}

void Teacher::backPropDis(Eigen::MatrixXf input, Eigen::MatrixXf desiredOutput, float step, float dx)
{
    Eigen::MatrixXf xnPartialDerivative = errorVector(mDiscriminator->process(input), desiredOutput, dx);

    propErrorDis(xnPartialDerivative, step);
}

void Teacher::propErrorDis(Eigen::MatrixXf xnPartialDerivative, float step)
{
    for(auto itr = mDiscriminator->rbegin(); itr != mDiscriminator->rend(); ++itr)
    {
        xnPartialDerivative = itr->backProp(xnPartialDerivative, step);
    }
}


Eigen::MatrixXf Teacher::errorVectorGen(Eigen::MatrixXf output, Eigen::MatrixXf desiredOutput, float dx)
{
    Eigen::MatrixXf errorVect = Eigen::MatrixXf::Zero(1, output.size());
    Eigen::MatrixXf discrOutput = mDiscriminator->process(output);

    for(unsigned int i(0); i < output.size(); ++i)
    {
        Eigen::MatrixXf deltaX(Eigen::MatrixXf::Zero(1, output.size()));
        deltaX(i) = dx;
        errorVect(i) = (mErrorFun(mDiscriminator->process(output + deltaX), desiredOutput) - mErrorFun(discrOutput, desiredOutput))/dx;
    }
    return errorVect;
}

Eigen::MatrixXf Teacher::errorVector(Eigen::MatrixXf output, Eigen::MatrixXf desiredOutput, float dx)
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
