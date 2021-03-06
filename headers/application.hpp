#ifndef APPLICATION_HPP
#define APPLICATION_HPP

#include <eigen3/Eigen/Dense>
#include <vector>
#include <random>

#include <headers/rapidjson/document.h>
#include "headers/neuralnetwork.hpp"
#include "headers/teacher.hpp"
#include "headers/inputOutput/inputprovider.hpp"
#include "headers/inputOutput/statscollector.hpp"
#include "headers/inputOutput/mnist_reader.h"
#include "headers/inputOutput/CSVFile.h"
#include "string.h"

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
///Classe destinée à gérer l'ensemble d'un projet
/**
 * La classe supervise l'apprentissage d'un réseau de neurones par rapport au batchs de données qu'on lui fournit
 * et sort les résultats dans un fichier csv
 */
class Application
{
    public:
        struct Config
        {
            float step;
            float dx;
			float sigmoidParameter;

            bool networkAreImported;
            bool useAverageForBatchlearning;

            unsigned int nbExperiments;
            unsigned int nbLoopsPerExperiment;
            unsigned int nbTeachingsPerLoop;
			unsigned int nbDisTeach;
			unsigned int nbGenTeach;
			unsigned int nbDisTest;
			unsigned int nbGenTest;
			unsigned int labelTrainSize;
			unsigned int labelTestSize;
            unsigned int intervalleImg;
            unsigned int nbImgParIntervalleImg;
			unsigned int minibatchSize;
            unsigned int genFunction;
            unsigned int descentTypeGen;
            unsigned int descentTypeDis;
            unsigned int imageSizeSide;

            std::string generatorPath;
            std::string discriminatorPath;
            std::string generatorDest;
            std::string discriminatorDest;
			std::string typeOfExperiment;
			std::string CSVFileNameImage;
			std::string CSVFileNameResult;
            std::string databaseToUse;

			
            std::vector<unsigned int> chiffresATracer;
            std::vector<std::string> classesCifar;

            std::vector<unsigned int> disLayerSizes;
            std::vector<unsigned int> genLayerSizes;

            std::vector<unsigned int> disLayerNbChannels;
            std::vector<unsigned int> genLayerNbChannels;

            std::vector<std::vector<unsigned int>> disLayerArgs;
            std::vector<std::vector<unsigned int>> genLayerArgs;

            std::vector<unsigned int> disLayerTypes;
            std::vector<unsigned int> genLayerTypes;
        };

    public:
        /// Un alias pour désigner un donnée (Entrée, Sortie)
        using Sample = std::pair<Eigen::MatrixXf, Eigen::MatrixXf>;
        /// Un alias pour désigner un batch de données (Entrée, Sortie)
        using Batch = std::vector<Sample>;
		/// Un alias pour désigner un minibatch de données (Entrée, Sortie)
		using Minibatch = Batch;

    public:
        /// Constructeur par batchs
        /**
         * Ce constructeur supervise le projet par rapport au réseau de neurones donné et aux batchs de tests et d'apprentissages donnés en paramètre
         */
        Application();

        /// Constructeur par fonction modèle
        /**
         * Ce constructeur supervise le projet par rapport au réseau de neurones donné, des batchs d'inputs pour l'apprentissage et les tests,
         * et la fonction à modéliser
         * @param network le réseau avec lequel on travaille
         * @param modelFunction la fonction à modéliser
         * @param teachingInputs les inputs pour l'apprentissage
         * @param testingInputs les inputs pour les tests
         */
        /*Application(NeuralNetwork::Ptr network,
                    std::function<Eigen::MatrixXf(Eigen::MatrixXf)> modelFunction,
                    std::vector<Eigen::MatrixXf> teachingInputs,
                    std::vector<Eigen::MatrixXf> testingInputs);*/

		void runExperiments();
	
        void runSingleStochasticExperiment();
	
		void runSingleMinibatchExperiment();

		void resetExperiment();
	

        /// Effectue une run d'apprentissage par méthode stochastique
        /**
         * Effectue une run d'apprentissage dont le nombre d'apprentissages est passé en paramètres
         */
        void runStochasticTeach();

		/// Effectue une run d'apprentissage par la méthode par batch
		/**
		 * Effectue une run d'apprentissage dont le nombre d'apprentissages est passé en paramètres
		 */
		void runMinibatchTeach();
	
        /// Effectue une run de tests sur D(G(z))
        /**
         * Effectue une run de test sur le batch de test
		 * @param limit  permet de limiter le nombre d'entrées de tests
		 * @param returnErrorRate deprecated
         */
        float runTestGen(int limit = -1, bool returnErrorRate = 1);

        /// Effectue une run de tests sur D(x)
        /**
         * Effectue une run de test sur le batch de test
		 * @param limit permet de limiter le nombre d'entrées de tests
		 * @param returnErrorRate deprecated
         */
        float runTestDis(int limit = -1, bool returnErrorRate = 1);

        /// Effectue une approximation du score des réseaux
        float gameScore(int nbImages);

        /// Génère une image à partir d'un input
        /**
         * Effectue un process de l'input par le Generateur
         * @param input un vecteur colonne, généralement, du bruit blanc
         */
        Eigen::MatrixXf genProcessing(Eigen::MatrixXf input);

	private:
		/// Génère un minibatch à partir d'un batch
		/**
		 * Génère un sous-ensemble du batch d'apprentissage ou du batch à partir de celui-ci
		 * @param batch le batch d'apprentissage ou le batch de test
		 */
		Minibatch sampleMinibatch(Batch batch);
	
	
		/// Génère un minibatch d'images obtenues par le générateur
		/**
		 * Génère un minibatch d'images obtenues par le générateur
		 */
		Minibatch sampleGeneratedImagesFromNoiseMinibatch();
	
//// Configuration


    private:
        /// Fonction pour charger la configuration de l'application
        void loadConfig(const std::string& configFileName = "config.json");
        void setConfig(rapidjson::Document& document);
        void exportPoids();
        NeuralNetwork* importNeuralNetwork(std::string networkPath, Functions::ActivationFun activationFun);

    private:

        /// Les réseaux avec lequel on travaille
        NeuralNetwork::Ptr  mDiscriminator;
        NeuralNetwork::Ptr  mGenerator;
        /// Le teacher qui permet de superviser l'apprentissage des réseaux
        Teacher             mTeacher;

        /// Le batch contenant tous les samples d'apprentissage du projet
        Batch               mTeachingBatchDis;
        /// Le batch contenant tous les samples de test du projet
        Batch               mTestingBatchDis;
        Batch               mTestingBatchGen;

        Stats::StatsCollector mStatsCollector;
        /// Un compteur permettant d'indicer les données exportées
        unsigned int        mTestCounter;

        /// Configuration de l'application
        Config              mConfig;
};

#endif // APPLICATION_HPP
