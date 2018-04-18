#ifndef HEADERSOISYLAYER_H
#define HEADERSOISYLAYER_H

#include "headers/fullconnectedlayer.hpp"

class NoisyLayer : public FullConnectedLayer
{
public:
    /// Constructeur permettant d'initialiser les paramètres de la couche de neurones
    /**
     * \param inputSize le nombre d'inputs de cette couche
     * \param outputSize le nombre d'outputs de cette couche
     * \param activationF la fonction d'activation de tous les neurones de la couche
     * \param descentType le type de descente utilisé dans l'apprentissage des fullConnectedLayers
     *
     * La matrice de poids est de dimension outputSize x inputSize
     */
    NoisyLayer(unsigned int inputSize, unsigned int outputSize, std::function<float(float)> activationF = Functions::sigmoid(10.f), unsigned int descentType = 0);


    /// La fonction effectuant le calcul de la sortie en fonction de l'entrée
    /**
     * \param inputs le vecteur d'input de la couche de neurones
     * \return le vecteur d'output de la couche de neurones
     * la fonction effectue le produit matriciel des poids par les entrées, puis applique la fonction d'activation
     */
    Eigen::MatrixXf processLayer(Eigen::MatrixXf inputs);


    /// La fonction effectuant les calculs de rétropropagation
    /**
     * La fonction calcule les 3 équations matricielles, mets à jour les poids et renvoie le vecteur de dérivées partielles
     * nécessaires pour la backprop de la couche précédente
     * @param xnPartialDerivative le vecteur des dérivées partielles selon Xn
     * @param step le pas d'apprentissage
     * @return le vecteur des dérivées partielles selon Xn-1 à envoyer à la couche précédente
     */
    Eigen::MatrixXf layerBackprop(Eigen::MatrixXf xnPartialDerivative, float step);

    /// La fonction effectuant la mise à jour des poids à la fin du Mini-Batch
    void            updateLayerWeights(unsigned int minibatchSize = 1);



private:
    /// La matrice de bruit liés à la couche
    Eigen::MatrixXf                 mNoiseWeight;

    /// Buffer pour stocker l'input de bruit, nécessaire pour la backprop
    Eigen::MatrixXf                 mBufferNoise;

    /// Buffer de la somme des variations des poids du bruit (nécessaire pour mini-batch)
    Eigen::MatrixXf					mSumNoiseVariation;

};

#endif // HEADERSOISYLAYER_H
