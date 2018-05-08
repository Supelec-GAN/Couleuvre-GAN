#include "headers/layers/zeropadlayer.hpp"

//*************CONSTRUCTEUR*************
//**************************************

ZeroPadLayer::ZeroPadLayer(unsigned int inputSize, unsigned int outputSize, unsigned int ZeroPadType)
: mInputDim((int) sqrt(inputSize))
, mOutputDim((int) sqrt(outputSize))
, mZeroPadType(ZeroPadType)
{
    if (inputSize > outputSize) throw "Size not matching for ZeroPadding (Input must be smaller than Output) !";
    mPropMatrix = Eigen::MatrixXf::Zero(inputSize, outputSize);
    mBackPropMatrix = Eigen::MatrixXf::Zero(outputSize, inputSize);
    int outputDimension = sqrt(outputSize);
    if (ZeroPadType == 0) //CLassique
    {
        mTailleZeroPadding= (((mOutputDim)-mInputDim)/2);
        for(int i(0); i < mInputDim; i++)
        {
            for(int j(0); j < mInputDim; j++)
            {
                mPropMatrix(i*(mInputDim)+j,mTailleZeroPadding*(outputDimension+1) + outputDimension*(i) + j) = 1;
                mBackPropMatrix(mTailleZeroPadding*(outputDimension+1) + outputDimension*(i) + j,i*(mInputDim)+j)=1;
            }
        }
    }
    else
    {
        if ((mOutputDim - mInputDim)/(mInputDim+1) != (int) (mOutputDim - mInputDim)/(mInputDim+1)) throw "Size not matching for Deconvolution !";
        mTailleZeroPadding = ((mOutputDim-mInputDim)/(mInputDim+1));
        for(int i(0); i < mInputDim; i++)
        {
            for(int j(0); j < mInputDim; j++)
            {
                mPropMatrix(i*(mInputDim)+j,mTailleZeroPadding*(mOutputDim+1) + (mTailleZeroPadding+1)*(mOutputDim)*(i) + j*(mTailleZeroPadding+1)) = 1;
                mBackPropMatrix(mTailleZeroPadding*(mOutputDim+1) + (mTailleZeroPadding+1)*(mOutputDim)*(i) + j*(mTailleZeroPadding+1),i*(mInputDim)+j)=1;
            }
        }
    }
}


ZeroPadLayer::~ZeroPadLayer(){}

//*************PROPAGATION**************
//**************************************

Eigen::MatrixXf ZeroPadLayer::processLayer(Eigen::MatrixXf inputs)
{
    return inputs*mPropMatrix;
}

//***********RETROPROPAGATION***********
//**************************************


Eigen::MatrixXf ZeroPadLayer::layerBackprop(Eigen::MatrixXf xnPartialDerivative, float step)
{
    return xnPartialDerivative*mBackPropMatrix;
}

[[deprecated]] //use minibatchLayerBackprop instead
Eigen::MatrixXf ZeroPadLayer::layerBackpropInvariant(Eigen::MatrixXf xnPartialDerivative)
{
    //Retour de x(n-1)PartialDerivative
    return xnPartialDerivative*mBackPropMatrix;
}

Eigen::MatrixXf ZeroPadLayer::minibatchLayerBackprop(Eigen::MatrixXf xnPartialDerivative, float step)
{
    return xnPartialDerivative*mBackPropMatrix;
}

//****************AUTRES****************
//**************************************

void ZeroPadLayer::updateLayerWeights(unsigned int minibatchSize)
{
    std::cout << "WARNING, useless function being used" << std::endl;
}

Eigen::MatrixXf ZeroPadLayer::fnDerivativeMatrix() const
{
    return Eigen::MatrixXf();
}


void ZeroPadLayer::reset()
{
}

int ZeroPadLayer::getInputSize()
{
    return (static_cast<int>(mInputDim*mInputDim));
}


/*std::ostream& operator<<(std::ostream& flux, ZeroPadLayer n)
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
