#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <list>
#include <vector>

#include "headers/functions.hpp"
#include "layers/fullconnectedlayer.hpp"
#include "layers/convolutionallayer.hpp"
#include "layers/noisylayer.h"
#include "layers/zeropadlayer.hpp"

class NeuralNetwork : public std::list<NeuronLayer::Ptr>
{
    public:
        using Ptr = std::shared_ptr<NeuralNetwork>;

    public:
	
	    /// Constructeur permettant d'initialiser une réseau neuronal vide
        NeuralNetwork();
	
        /// Constructeur permettant d'initialiser un réseau neuronal avec choix des fonctions d'activation
        /**
         * Constructeur permettant l'initialisation d'un réseau à n couches à partir des (n+1) tailles d'input/output
         * (la sortie d'une couche est l'entrée de la suivante), avec choix des fonctions d'activation
         * @param layerSizes les tailles des vecteurs d'entrées/sorties
         * @param activationFuns le vector contenant les fonctions d'activation de chaque couche
         * @param descentType le type de descente utilisé dans l'apprentissage des fullConnectedLayers
         */

        NeuralNetwork(std::vector<unsigned int> layerTypes, std::vector<unsigned int> layerSizes, std::vector<unsigned int> layerChannels, std::vector<std::vector<unsigned int>> layerArgs, std::vector<Functions::ActivationFun> activationFuns, unsigned int descentType = 0);

        /// Constructeur permettant d'initialiser un réseau neuronal avec choix des fonctions d'activation et des poids donnés
        /**
         * Constructeur permettant l'initialisation d'un réseau à n couches à partir des (n+1) tailles d'input/output
         * (la sortie d'une couche est l'entrée de la suivante), avec choix des fonctions d'activation
         * @param activationFuns le vector contenant les fonctions d'activation de chaque couche
         * @param descentType le type de descente utilisé dans l'apprentissage des fullConnectedLayers
         */

        NeuralNetwork(std::vector<unsigned int> layerSizes, std::vector<Eigen::MatrixXf> weightVector, std::vector<Eigen::MatrixXf> biasVector, std::vector<Functions::ActivationFun> activationFuns, unsigned int descentType = 0);

        /// Constructeur permettant d'initialiser un réseau neuronal avec la fonction par défaut et une descente normale
        /**
         * Constructeur permettant l'initialisation d'un réseau à n couches à partir des (n+1) tailles d'input/output
         * (la sortie d'une couche est l'entrée de la suivante). La fonction d'activation choisie est la fonction d'activation par défaut
         * @param layerSizes les tailles des vecteurs d'entrées/sorties
         */

        NeuralNetwork(std::vector<unsigned int> layerSizes);

		/// Constructeur permettant d'initialiser le réseau neuronal avec un conteneur (vector, list...) de neuronLayer
		/**
		 * @param layerList la liste des couches de neurones
		 */
		template <typename Container>
		NeuralNetwork(Container layerList);
	


        Eigen::MatrixXf processNetwork(Eigen::MatrixXf input);
        Eigen::MatrixXf processNetwork();

		void reset();
	
        int getInputSize();


    public:
        /// Fonction utilitaire permettant d'afficher le réseau de neurones
        /**
         * Cette fonction affiche les matrices de poids des différents layers du réseau
         */
        friend std::ostream& operator<<(std::ostream& flux, NeuralNetwork network);

};

#include "headers/neuralnetwork.inl"

#endif // NEURALNETWORK_HPP
