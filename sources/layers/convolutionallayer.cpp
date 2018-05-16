#include "headers/layers/convolutionallayer.hpp"


//*************CONSTRUCTEUR*************
//**************************************

ConvolutionalLayer::ConvolutionalLayer(unsigned int tailleImg, unsigned int nbChannels, unsigned int dimensionFiltre, unsigned int nbFiltres, std::function<float(float)> activationF)
: mDimensionInput((int) sqrt(tailleImg))
, mWeight(std::vector<Eigen::MatrixXf>())
, mBias(Eigen::MatrixXf::Random(1, nbFiltres)) 				//ligne
, mActivationFun(activationF)
, mBufferActivationLevel(Eigen::MatrixXf::Zero(nbFiltres, (mDimensionInput-dimensionFiltre + 1)*(mDimensionInput-dimensionFiltre + 1)))	//ligne
, mBufferInput(Eigen::MatrixXf::Zero(nbChannels, tailleImg))				//ligne
, mSumWeightVariation(std::vector<Eigen::MatrixXf>())
, mSumBiasVariation(Eigen::MatrixXf::Zero(1, nbFiltres))
, mInputDimension(mDimensionInput)
, mInputChannels(nbChannels)
{
    //Matrice de poids
    for(int i(0); i < nbFiltres; i++)
    {
        mWeight.push_back(Eigen::MatrixXf::Random(nbChannels,dimensionFiltre*dimensionFiltre));
        mSumWeightVariation.push_back(Eigen::MatrixXf::Zero(nbChannels,dimensionFiltre*dimensionFiltre));
    }
}

ConvolutionalLayer::ConvolutionalLayer(unsigned int tailleImg, unsigned int nbChannels, std::vector<Eigen::MatrixXf> weight, std::function<float(float)> activationF)
: mDimensionInput((int) sqrt(tailleImg))
, mWeight(weight)
, mBias(Eigen::MatrixXf::Random(1, weight.size())) 				//ligne
, mActivationFun(activationF)
, mBufferActivationLevel(Eigen::MatrixXf::Zero(weight.size(), (mDimensionInput-(int)sqrt(weight[0].cols()) + 1)*(mDimensionInput-(int)sqrt(weight[0].cols()) + 1)))	//ligne
, mBufferInput(Eigen::MatrixXf::Zero(nbChannels, tailleImg))				//ligne
, mSumWeightVariation(std::vector<Eigen::MatrixXf>())
, mSumBiasVariation(Eigen::MatrixXf::Zero(1, weight.size()))
, mInputDimension(mDimensionInput)
, mInputChannels(nbChannels)
{
    //Matrice de poids
    for(int i(0); i < weight.size(); i++)
    {   for(int j(0); j< weight[0].cols();j++)
            mWeight[i](0,j) = weight[i](0,j);
        mSumWeightVariation.push_back(Eigen::MatrixXf::Zero(nbChannels,weight[0].cols()));
    }
}

ConvolutionalLayer::~ConvolutionalLayer(){}

//*************PROPAGATION**************
//**************************************

Eigen::MatrixXf ConvolutionalLayer::processLayer(Eigen::MatrixXf inputs)
{
    mBufferInput = inputs;
    if (inputs.cols()!=mInputDimension*mInputDimension) throw;
    for (int n=0; n < mWeight.size(); n++) //Pour chaque filtre...
    {
        auto temp = ConvolutionalLayer::convolution(inputs, mWeight[n], true);
        mBufferActivationLevel.row(n) = temp;
    }
    Eigen::MatrixXf output = mBufferActivationLevel;

    for(unsigned int i(0); i < output.cols(); i++)
        for(unsigned int j(0); j < output.rows(); j++)
            output(j,i) = mActivationFun(output(j,i));

    return output;
}

//***********RETROPROPAGATION***********
//**************************************


Eigen::MatrixXf ConvolutionalLayer::layerBackprop(Eigen::MatrixXf xnPartialDerivative, float step)
{
    // Calcul de ynPartialDerivative
    Eigen::MatrixXf ynPartialDerivative = xnPartialDerivative*fnDerivativeMatrix();

    int weightDimension = sqrt(mWeight[0].cols());
    int ynDerivDimension = sqrt(ynPartialDerivative.cols());

    //Mise à jour des poids
    for(int i(0); i < mSumWeightVariation.size(); i++) //Pour chaque filtre...
    {
        Eigen::MatrixXf ynPartialDerivativeCarree = Eigen::MatrixXf::Ones(mBufferInput.rows(),1)*ynPartialDerivative.row(i);
        mSumWeightVariation[i] += step*ConvolutionalLayer::convolution(mBufferInput,ynPartialDerivativeCarree, false);
    }
    //mSumBiasVariation += step*ynPartialDerivative;

    //Retour de x(n-1)PartialDerivative
    int incrementDimension = 2*(weightDimension-1);
    int zeroPaddingDimension = incrementDimension+ynDerivDimension;

    Eigen::MatrixXf ynZeroPadding = Eigen::MatrixXf::Zero(ynPartialDerivative.rows(),zeroPaddingDimension*zeroPaddingDimension);
    for (int n=0; n < ynPartialDerivative.rows(); n++) //Pour chaque channel...
    {
        for (int i=0; i < ynDerivDimension; i++)
        {
            for (int j=0; j < ynDerivDimension; j++)
            {
                ynZeroPadding(n, (i+incrementDimension/2)*zeroPaddingDimension + j + incrementDimension/2) = ynPartialDerivative(n,i*ynDerivDimension+j);
            }
        }
    }
    Eigen::MatrixXf resultat = Eigen::MatrixXf::Zero(mInputChannels, mBufferInput.cols()); //mBufferInput.cols() est la taille de l'input
    for (int i=0; i < mWeight.size(); i++) //Pour tous les channels
    {
        Eigen::MatrixXf ynZeroPaddingCarree = Eigen::MatrixXf::Ones(mWeight[i].rows(),ynZeroPadding.rows())*ynZeroPadding; //On somme l'ensemble des erreurs d'une ligne pour tous les channels
        resultat += convolution(ynZeroPaddingCarree,mWeight[i].reverse(), false);
    }
    updateLayerWeights();
    return resultat;
}
/*
Eigen::MatrixXf ConvolutionalLayer::layerBackprop(Eigen::MatrixXf xnPartialDerivative, float step)
{
    // Calcul de ynPartialDerivative
    Eigen::MatrixXf ynPartialDerivative = xnPartialDerivative*fnDerivativeMatrix();

    //Mise à jour des poids
    Eigen::MatrixXf wnPartialDerivative = (mBufferInput.transpose())*ynPartialDerivative;
    mSumBiasVariation += step*ynPartialDerivative;
    mSumWeightVariation += step*wnPartialDerivative;
    updateLayerWeights();

    //Retour de x(n-1)PartialDerivative
    return ynPartialDerivative*mWeight.transpose();
}*/

[[deprecated]] //use minibatchLayerBackprop instead
Eigen::MatrixXf ConvolutionalLayer::layerBackpropInvariant(Eigen::MatrixXf xnPartialDerivative)
{
    // Calcul de ynPartialDerivative
    Eigen::MatrixXf ynPartialDerivative = xnPartialDerivative*fnDerivativeMatrix();

    int weightDimension = sqrt(mWeight[0].cols());
    int ynDerivDimension = sqrt(ynPartialDerivative.cols());

    //Retour de x(n-1)PartialDerivative
    int incrementDimension = 2*(weightDimension-1);
    int zeroPaddingDimension = incrementDimension+ynDerivDimension;

    Eigen::MatrixXf ynZeroPadding = Eigen::MatrixXf::Zero(ynPartialDerivative.rows(),zeroPaddingDimension*zeroPaddingDimension);
    for (int n=0; n < ynPartialDerivative.rows(); n++) //Pour chaque channel...
    {
        for (int i=0; i < ynDerivDimension; i++) //Copie du symmétrique de ynPartialDerivative dans le Zeropadding (A_i,j = A_n-i,m-j)
        {
            for (int j=0; j < ynDerivDimension; j++)
            {
                ynZeroPadding(n, (zeroPaddingDimension+1)*incrementDimension + j + i*zeroPaddingDimension) = ynPartialDerivative(n,(ynDerivDimension-i-1)*ynDerivDimension + (ynDerivDimension-j-1));
            }
        }
    }
    Eigen::MatrixXf resultat = Eigen::MatrixXf::Zero(mInputChannels, mBufferInput.cols()); //mBufferInput.cols() est la taille de l'input
    for (int i=0; i < mWeight.size(); i++)
    {
        Eigen::MatrixXf ynZeroPaddingCarree = Eigen::MatrixXf::Ones(mWeight[i].rows(),ynZeroPadding.rows())*ynZeroPadding;
        resultat += convolution(ynZeroPaddingCarree,mWeight[i], false);
    }
    return resultat;
}

Eigen::MatrixXf ConvolutionalLayer::minibatchLayerBackprop(Eigen::MatrixXf xnPartialDerivative, float step)
{
    // Calcul de ynPartialDerivative
    Eigen::MatrixXf ynPartialDerivative = xnPartialDerivative*fnDerivativeMatrix();

    int weightDimension = sqrt(mWeight[0].cols());
    int ynDerivDimension = sqrt(ynPartialDerivative.cols());

    //Mise à jour des poids
    for(int i(0); i < mSumWeightVariation.size(); i++) //Pour chaque filtre...
    {
        Eigen::MatrixXf ynPartialDerivativeCarree = Eigen::MatrixXf::Ones(mBufferInput.rows(),1)*ynPartialDerivative.row(i);
        mSumWeightVariation[i] += step*ConvolutionalLayer::convolution(mBufferInput,ynPartialDerivativeCarree, false);
    }
    //mSumBiasVariation += step*ynPartialDerivative;

    //Retour de x(n-1)PartialDerivative
    int incrementDimension = 2*(weightDimension-1);
    int zeroPaddingDimension = incrementDimension+ynDerivDimension;

    Eigen::MatrixXf ynZeroPadding = Eigen::MatrixXf::Zero(ynPartialDerivative.rows(),zeroPaddingDimension*zeroPaddingDimension);
    for (int n=0; n < ynPartialDerivative.rows(); n++) //Pour chaque channel...
    {
        for (int i=0; i < ynDerivDimension; i++) //Copie du symmétrique de ynPartialDerivative dans le Zeropadding (A_i,j = A_n-i,m-j)
        {
            for (int j=0; j < ynDerivDimension; j++)
            {
                ynZeroPadding(n, (zeroPaddingDimension+1)*incrementDimension + j + i*zeroPaddingDimension) = ynPartialDerivative(n,(ynDerivDimension-i-1)*ynDerivDimension + (ynDerivDimension-j-1));
            }
        }
    }
    Eigen::MatrixXf resultat = Eigen::MatrixXf::Zero(mInputChannels, mBufferInput.cols()); //mBufferInput.cols() est la taille de l'input
    for (int i=0; i < mWeight.size(); i++)
    {
        Eigen::MatrixXf ynZeroPaddingCarree = Eigen::MatrixXf::Ones(mWeight[i].rows(),ynZeroPadding.rows())*ynZeroPadding;
        resultat += convolution(ynZeroPaddingCarree,mWeight[i], false);
    }
    return resultat;
}

//****************AUTRES****************
//**************************************

void ConvolutionalLayer::updateLayerWeights(unsigned int minibatchSize)
{
    for (int i(0); i<mWeight.size(); i++)
    {
        mWeight[i] -= mSumWeightVariation[i]/minibatchSize;
        mSumWeightVariation[i].setZero() ;
    }
    mBias += mSumBiasVariation/minibatchSize;

    //reset des buffer
    mSumBiasVariation.setZero();
}

Eigen::MatrixXf ConvolutionalLayer::fnDerivativeMatrix() const
{
    auto fnDerivated = [this] (float x, float dx)
                        {
                            return (mActivationFun(x+dx) - mActivationFun(x))/dx;
                        };

    Eigen::MatrixXf fnDerivativeMat(mBufferActivationLevel.cols(),1);
    for(auto i(0); i < mBufferActivationLevel.cols(); ++i)
        fnDerivativeMat(i) = fnDerivated(mBufferActivationLevel(i), 0.05);

    return Eigen::MatrixXf(fnDerivativeMat.asDiagonal());
}

void ConvolutionalLayer::reset()
{
    for (int i(0); i < mWeight.size(); i++)
        mWeight[i] = Eigen::MatrixXf::Random(mWeight[i].rows(), mWeight[i].cols());
    mBias = Eigen::MatrixXf::Random(1,mBias.cols());
}

int ConvolutionalLayer::getInputSize()
{
    return (static_cast<int>(mDimensionInput*mDimensionInput));
}

Eigen::MatrixXf ConvolutionalLayer::convolution(Eigen::MatrixXf input, Eigen::MatrixXf filtre, bool sommerLignes)
{
    std::vector<std::thread> threads;
    int inputDimension = sqrt(input.cols());
    int filtreDimension = sqrt(filtre.cols());
    int nbChannels = filtre.rows();
    if (sommerLignes) nbChannels = 1;
    std::shared_ptr<Eigen::MatrixXf> resultat(new Eigen::MatrixXf(Eigen::MatrixXf::Zero(nbChannels, (inputDimension - filtreDimension+1)*(inputDimension - filtreDimension+1))));
    Convolution conv(input, filtre, resultat, sommerLignes);
    for (int n=0; n < filtre.rows(); n++) //Pour chaque channel...
    {
        threads.push_back(std::thread(conv, n));
    }
    //std::cout << "synchronizing all threads...\n";
    for (auto& th : threads) th.join();
    return(*resultat);
}

Eigen::MatrixXf ConvolutionalLayer::convolutionMonothreade(Eigen::MatrixXf input, Eigen::MatrixXf filtre, bool sommerLignes)
{
    int inputDimension = sqrt(input.cols());
    int filtreDimension = sqrt(filtre.cols());
    Eigen::MatrixXf filtreConv = Eigen::MatrixXf::Zero(input.cols(), (inputDimension - filtreDimension+1)*(inputDimension - filtreDimension+1));
    Eigen::MatrixXf resultat;
    if (sommerLignes)
        resultat = Eigen::MatrixXf::Zero(1, (inputDimension - filtreDimension+1)*(inputDimension - filtreDimension+1));
    else
        resultat = Eigen::MatrixXf::Zero(filtre.rows(), (inputDimension - filtreDimension+1)*(inputDimension - filtreDimension+1));
    for (int n=0; n < filtre.rows(); n++) //Pour chaque channel...
    {
        for (int k=0; k< inputDimension-filtreDimension+1; k++) //Pour chaque décalage vertical du filtre...
        {
            for (int l=0; l< inputDimension-filtreDimension+1; l++) //Pour chaque décalage latéral du filtre...
            {
                for (int i=0; i< filtreDimension; i++) //Parcours des lignes du filtre...
                {
                    for (int j=0; j< filtreDimension; j++) //Parcours des colonnes du filtre...
                    {
                        filtreConv(l+j+(i+k)*inputDimension,l+k*(inputDimension-filtreDimension+1)) = filtre(n,i*filtreDimension+j); //On crée une matrice de poids qui va nous permettre de lancer un calcul qui sera parallélisable
                    }
                }
            }
        }
        if (sommerLignes)
            resultat += input.row(n)*filtreConv;
        else
            resultat.row(n) = input.row(n)*filtreConv;
        filtreConv.setZero();
    }
    return(resultat);
}


/*std::ostream& operator<<(std::ostream& flux, ConvolutionalLayer n)
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
