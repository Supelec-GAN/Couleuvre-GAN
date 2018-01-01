#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <list>
#include <vector>

#include "neuronlayer.hpp"
#include "headers/functions.hpp"

class NeuralNetwork : public std::list<NeuronLayer>
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
         */
        NeuralNetwork(std::vector<unsigned int> layerSizes, std::vector<Functions::ActivationFun> activationFuns);
	
        /// Constructeur permettant d'initialiser un réseau neuronal avec la fonction par défaut
        /**
         * Constructeur permettant l'initialisation d'un réseau à n couches à partir des (n+1) tailles d'input/output
         * (la sortie d'une couche est l'entrée de la suivante). La fonction d'activation choisie est la fonction d'activation par défaut
         * @param layerSizes les tailles des vecteurs d'entrées/sorties
         */
        NeuralNetwork(std::vector<unsigned int> layerSizes);

        void reset();

		/// Constructeur permettant d'initialiser le réseau neuronal avec un conteneur (vector, list...) de neuronLayer
		/**
		 * @param layerList la liste des couches de neurones
		 */
        template <typename Container>
        NeuralNetwork(Container layerList);

        Eigen::MatrixXf process(Eigen::MatrixXf input);
        Eigen::MatrixXf process();

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
