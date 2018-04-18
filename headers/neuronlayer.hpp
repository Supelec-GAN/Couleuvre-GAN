#ifndef NEURONLAYER_HPP
#define NEURONLAYER_HPP

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <functional>
#include <memory>

#include "headers/functions.hpp"

/// Classe modélisant une couche de neurones
/**
 *  NeuroneLayer représente une couche de neurones, avec une matrice de poids et une fonction d'activation
 */
class NeuronLayer
{
        public:
            using Ptr = std::unique_ptr<NeuronLayer>;
        public :

        virtual ~NeuronLayer();

        /// La fonction effectuant le calcul de la sortie en fonction de l'entrée
        /**
         * \param inputs le vecteur d'input de la couche de neurones
         * \return le vecteur d'output de la couche de neurones
         * la fonction effectue le produit matriciel des poids par les entrées, puis applique la fonction d'activation
         */
        virtual Eigen::MatrixXf processLayer(Eigen::MatrixXf inputs) =0;

        /// La fonction effectuant les calculs de rétropropagation
        /**
         * La fonction calcule les 3 équations matricielles, mets à jour les poids et renvoie le vecteur de dérivées partielles
         * nécessaires pour la backprop de la couche précédente
         * @param xnPartialDerivative le vecteur des dérivées partielles selon Xn
         * @param step le pas d'apprentissage
         * @return le vecteur des dérivées partielles selon Xn-1 à envoyer à la couche précédente
         */
        virtual Eigen::MatrixXf layerBackprop(Eigen::MatrixXf xnPartialDerivative, float step) =0;

	
        /// La fonction effectuant les calculs de rétropropagation sans mise à jour du réseau
        /**
         * La fonction propage l'erreur comme pour backprop, mais ne change pas les poids et biais. On ne définit donc pas de pas d'apprentissage
         * @param xnPartialDerivative le vecteur des dérivées partielles selon Xn
         * @return le vecteur des dérivées partielles selon Xn-1 à envoyer à la couche précédente
         */
        virtual Eigen::MatrixXf layerBackpropInvariant(Eigen::MatrixXf xnPartialDerivative) =0;
	
		/// La fonction effectuant les calculs de rétropropagation sans mise à jour du réseau selon le principe du mini-batch
		/**
		 * La fonction calcule les 3 équations matricielles, somme les poids à modifier et
		 * renvoie le vecteur de dérivées partielles
		 * nécessaire pour la backprop de la couche précédente
		 * @param xnPartialDerivative le vecteur des dérivées partielles selon Xn
		 * @param step le pas d'apprentissage
		 * @return le vecteur des dérivées partielles selon Xn-1 à envoyer à la couche précédente
		 */
        virtual Eigen::MatrixXf minibatchLayerBackprop(Eigen::MatrixXf xnPartialDerivative, float step) =0;


		/// La fonction effectuant la mise à jour des poids à la fin du Mini-Batch
        virtual void            updateLayerWeights(unsigned int minibatchSize = 1) =0;

        virtual void            reset() =0;

        virtual int             getInputSize() =0;

    public:
        /// Fonction utilitaire permettant d'afficher le neurone
        /**
         * Cette fonction affiche la matrice des poids
         */
        //friend std::ostream& operator<<(std::ostream& flux, NeuronLayer nl);

    private:
        /// Fonction renvoyant le vecteur des dérivées de Fn évalué en Yn
        /**
         * Cette fonction calcule Fn'(Yn) ou Yn = mBufferActivationLevel
         * @return le vecteur des dérivées mises en colonne
         */

        virtual Eigen::MatrixXf fnDerivativeMatrix() const =0;
};

#endif // NEURONLAYER_HPP
