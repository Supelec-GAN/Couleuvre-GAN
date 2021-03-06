#ifndef ZEROPADLAYER_HPP
#define ZEROPADLAYER_HPP

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <functional>
#include <memory>
#include "headers/functions.hpp"
#include "headers/neuronlayer.hpp"

class NeuronLayer;
/// Classe modélisant une couche de neurones
/**
 *  NeuroneLayer représente une couche de neurones, avec une matrice de poids et une fonction d'activation
 */
class ZeroPadLayer : public NeuronLayer
{
    public:
        /// Constructeur permettant d'initialiser les paramètres de la couche de neurones
        /**
         * \param inputSize le nombre d'inputs de cette couche
         * \param outputSize le nombre d'outputs de cette couche
         * \param descentType le type de descente utilisé dans l'apprentissage des ZeroPadLayers
         *
         * La matrice de poids est de dimension outputSize x inputSize
         */
                        ZeroPadLayer(unsigned int inputSize, unsigned int outputSize, unsigned int descentType = 0);

                        ~ZeroPadLayer();

        /// La fonction effectuant le calcul de la sortie en fonction de l'entrée
        /**
         * \param inputs le vecteur d'input de la couche de neurones
         * \return le vecteur d'output de la couche de neurones
         * la fonction effectue le produit matriciel des poids par les entrées, puis applique la fonction d'activation
         */
        virtual Eigen::MatrixXf processLayer(Eigen::MatrixXf inputs);

        /// La fonction effectuant les calculs de rétropropagation
        /**
         * La fonction calcule les 3 équations matricielles, mets à jour les poids et renvoie le vecteur de dérivées partielles
         * nécessaires pour la backprop de la couche précédente
         * @param xnPartialDerivative le vecteur des dérivées partielles selon Xn
         * @param step le pas d'apprentissage
         * @return le vecteur des dérivées partielles selon Xn-1 à envoyer à la couche précédente
         */
        virtual Eigen::MatrixXf layerBackprop(Eigen::MatrixXf xnPartialDerivative, float step);


        /// La fonction effectuant les calculs de rétropropagation sans mise à jour du réseau
        /**
         * La fonction propage l'erreur comme pour backprop, mais ne change pas les poids et biais. On ne définit donc pas de pas d'apprentissage
         * @param xnPartialDerivative le vecteur des dérivées partielles selon Xn
         * @return le vecteur des dérivées partielles selon Xn-1 à envoyer à la couche précédente
         */
        Eigen::MatrixXf layerBackpropInvariant(Eigen::MatrixXf xnPartialDerivative);

        /// La fonction effectuant les calculs de rétropropagation sans mise à jour du réseau selon le principe du mini-batch
        /**
         * La fonction calcule les 3 équations matricielles, somme les poids à modifier et
         * renvoie le vecteur de dérivées partielles
         * nécessaire pour la backprop de la couche précédente
         * @param xnPartialDerivative le vecteur des dérivées partielles selon Xn
         * @param step le pas d'apprentissage
         * @return le vecteur des dérivées partielles selon Xn-1 à envoyer à la couche précédente
         */
        Eigen::MatrixXf minibatchLayerBackprop(Eigen::MatrixXf xnPartialDerivative, float step);


        /// La fonction effectuant la mise à jour des poids à la fin du Mini-Batch
        virtual void    updateLayerWeights(unsigned int minibatchSize = 1);

        void            reset();

        int             getInputSize();

    public:
        /// Fonction utilitaire permettant d'afficher le neurone
        /**
         * Cette fonction affiche la matrice des poids
         */
        //friend std::ostream& operator<<(std::ostream& flux, ZeroPadLayer nl);

    protected:
        /// Fonction renvoyant le vecteur des dérivées de Fn évalué en Yn
        /**
         * Cette fonction calcule Fn'(Yn) ou Yn = mBufferActivationLevel
         * @return le vecteur des dérivées mises en colonne
         */
        Eigen::MatrixXf fnDerivativeMatrix() const;


    protected:
        /// La matrice des poids de la couche de neurones
        Eigen::MatrixXf                 mPropMatrix;

        /// La matrice des biais de la couche de neurones
        Eigen::MatrixXf                 mBackPropMatrix;

        unsigned int                    mInputDim;

        unsigned int                    mOutputDim;

        unsigned int                    mTailleZeroPadding;

        /// Int déterminant le type de descente dans l'apprentissage
        unsigned int                    mZeroPadType;

};

#endif // ZEROPADLAYER_HPP
