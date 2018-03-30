#ifndef CONVOLUTIONALLAYER_HPP
#define CONVOLUTIONALLAYER_HPP

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <functional>
#include <memory>
#include <vector>

#include "headers/functions.hpp"
#include "neuronlayer.hpp"

class NeuronLayer;


/// Classe modélisant une couche de neurones
/**
 *  NeuroneLayer représente une couche de neurones, avec une matrice de poids et une fonction d'activation
 */
class ConvolutionalLayer : public NeuronLayer
{
    public:
        /// Constructeur permettant d'initialiser les paramètres de la couche de neurones
        /**
         * \param inputSize le nombre d'inputs de cette couche
         * \param outputSize le nombre d'outputs de cette couche
         * \param activationF la fonction d'activation de tous les neurones de la couche
         *
         * La matrice de poids est de dimension outputSize x inputSize
         */
                        ConvolutionalLayer(unsigned int inputSize, unsigned int nbChannels, unsigned int dimensionFiltre, unsigned int nbFiltres, std::function<float(float)> activationF = Functions::sigmoid(10.f));

        /// Constructeur permettant d'initialiser les paramètres de la couche de neurones
        /**
         * \param inputSize le nombre d'inputs de cette couche
         * \param outputSize le nombre d'outputs de cette couche
         * \param activationF la fonction d'activation de tous les neurones de la couche
         *
         * La matrice de poids est de dimension outputSize x inputSize
         */
                        //ConvolutionalLayer(unsigned int inputSize, unsigned int nbFiltres, unsigned int taille, std::vector<Eigen::MatrixXf> weight, Eigen::MatrixXf bias, std::function<float(float)> activationF = Functions::sigmoid(10.f));

                        ~ConvolutionalLayer();

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
        void            updateLayerWeights(unsigned int minibatchSize = 1);

        void            reset();

        int             getInputSize();

        static Eigen::MatrixXf convolution(Eigen::MatrixXf input, Eigen::MatrixXf filtre, bool sommerLignes = true);

    public:
        /// Fonction utilitaire permettant d'afficher le neurone
        /**
         * Cette fonction affiche la matrice des poids
         */
        //friend std::ostream& operator<<(std::ostream& flux, ConvolutionalLayer nl);

    private:
        /// Fonction renvoyant le vecteur des dérivées de Fn évalué en Yn
        /**
         * Cette fonction calcule Fn'(Yn) ou Yn = mBufferActivationLevel
         * @return le vecteur des dérivées mises en colonne
         */
        Eigen::MatrixXf fnDerivativeMatrix() const;

    private:

        unsigned int                    mDimensionInput;

        /// La matrice des poids de la couche de neurones
        std::vector<Eigen::MatrixXf>    mWeight;

        /// La matrice des biais de la couche de neurones
        Eigen::MatrixXf                 mBias;

        /// La fonction d'activation de la couche de neurones
        std::function<float(float)>     mActivationFun;

        /// Buffer pour stocker Yn = WnXn-1, nécessaire pour la backprop
        Eigen::MatrixXf                 mBufferActivationLevel;

        /// Buffer pour stocker l'input, nécessaire pour la backrprop
        Eigen::MatrixXf                 mBufferInput;

        /// Buffer de la somme des variations du biais au sein d'un mini-batch
        std::vector<Eigen::MatrixXf>    mSumWeightVariation;

        /// Buffer de la somme des variation de poids au sein d'un mini-batch
        Eigen::MatrixXf 				mSumBiasVariation;

        unsigned int                    mInputDimension;

        unsigned int                    mInputChannels;


};

#endif // CONVOLUTIONALLAYER_HPP
