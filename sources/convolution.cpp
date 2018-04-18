#include "headers/convolution.hpp"

std::mutex mtx;

Convolution::Convolution(Eigen::MatrixXf input, Eigen::MatrixXf filtre, std::shared_ptr<Eigen::MatrixXf> resultat, bool sommerLignes)
: mInput(input)
, mFiltre(filtre)
, mResultat(resultat)
, mSommerLignes(sommerLignes)
{}

void Convolution::operator()(int id)
{
    int inputDimension = sqrt(mInput.cols());
    int filtreDimension = sqrt(mFiltre.cols());
    Eigen::MatrixXf filtreConv = Eigen::MatrixXf::Zero(mInput.cols(), (inputDimension - filtreDimension+1)*(inputDimension - filtreDimension+1));
    for (int k=0; k< inputDimension-filtreDimension+1; k++) //Pour chaque décalage vertical du filtre...
    {
        for (int l=0; l< inputDimension-filtreDimension+1; l++) //Pour chaque décalage latéral du filtre...
        {
            for (int i=0; i< filtreDimension; i++) //Parcours des lignes du filtre...
            {
                for (int j=0; j< filtreDimension; j++) //Parcours des colonnes du filtre...
                {
                    filtreConv(l+j+(i+k)*inputDimension,l+k*(inputDimension-filtreDimension+1)) = (mFiltre)(id,i*filtreDimension+j); //On crée une matrice de poids qui va nous permettre de lancer un calcul qui sera parallélisable
                }
            }
        }
    }
    Eigen::MatrixXf temp = mInput.row(id)*filtreConv;
    mtx.lock();
    if (mSommerLignes)
        (*mResultat) += temp;
    else
        (*mResultat).row(id) = temp;
    mtx.unlock();
}

