#ifndef APPLICATION_HPP
#define APPLICATION_HPP

#include <eigen3/Eigen/Dense>
#include <vector>
#include <random>

#include <headers/rapidjson/document.h>
#include "headers/neuralnetwork.hpp"
#include "headers/teacher.hpp"
#include "headers/statscollector.hpp"

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
        };

    public:
        /// Un alias pour désigner un donnée (Entrée, Sortie)
        using Sample = std::pair<Eigen::MatrixXf, Eigen::MatrixXf>;
        /// Un alias pour désigner un batch de données (Entrée, Sortie)
        using Batch = std::vector<Sample>;

    public:
        /// Constructeur par batchs
        /**
         * Ce constructeur supervise le projet par rapport au réseau de neurones donné et aux batchs de tests et d'apprentissages donnés en paramètre
         * @param discriminator le discriminateur avec lequel on travaille
		 * @param generator le generateur avec lequel on travaille
         * @param teachingBatch le batch des données servant à l'apprentissage
         * @param testBatch le batch des données de test
         */
        Application(NeuralNetwork::Ptr discriminator, NeuralNetwork::Ptr generator, Batch teachingBatch, Batch testBatch);

        /// Constructeur par fonction modèle
        /**
         * Ce constructeur supervise le projet par rapport au réseaude neurones donné, des batchs d'inputs pour l'apprentissage et les tests,
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

#pragma mark - Expériences
	
		void runExperiments(unsigned int nbExperiments, unsigned int nbLoops, unsigned int nbTeachingsPerLoop);
	
		void runSingleStochasticExperiment(unsigned int nbLoops, unsigned int nbTeachingsPerLoop);
	
		void resetExperiment();
	
#pragma mark - Apprentissage

        /// Effectue une run d'apprentissage par méthode stochastique
        /**
         * Effectue une run d'apprentissage dont le nombre d'apprentissages est passé en paramètres
         * @param nbTeachings le nombre d'apprentissages à faire pendant la run
         */
        void runStochasticTeach(unsigned int nbTeachings, bool trigger);

		/// Effectue une run d'apprentissage par la méthode par batch
		/**
		 * Effectue une run d'apprentissage dont le nombre d'apprentissages est passé en paramètres
		 * @param nbTeachings le nombre d'apprentissages à faire pendant la run
		 */
		void runMiniBatchTeach(unsigned int nbTeachings, unsigned int batchSize);
	
        /// Effectue une run de tests
        /**
         * Effectue une run de test sur le batch de test
         */
        float runTest(int limit = -1, bool returnErrorRate = 1);

        /// Effectue une approximation du score des réseaux
        float gameScore(int nbImages);

        /// Génère une image à partir d'un input
        /**
         * Effectue un process de l'input par le Generateur
         * @param input un vecteur colonne, généralement, du bruit blanc
         */
        Eigen::MatrixXf genProcessing(Eigen::MatrixXf input);
	
#pragma mark - Configuration

    private:
        /// Fonction pour charger la configuration de l'application
        void loadConfig(const std::string& configFileName = "config.json");
        void setConfig(rapidjson::Document& document);

    private:
        /// Les réseaux avec lequel on travaille
        NeuralNetwork::Ptr  mDiscriminator;
        NeuralNetwork::Ptr  mGenerator;
        /// Le teacher qui permet de superviser l'apprentissage des réseaux
        Teacher             mTeacher;

        /// Le batch contenant tous les samples d'apprentissage du projet
        Batch               mTeachingBatch;
        /// Le batch contenant tous les samples de test du projet
        Batch               mTestingBatch;

        Stats::StatsCollector mStatsCollector;
        /// Un compteur permettant d'indicer les données exportées
        unsigned int        mTestCounter;

        /// Configuration de l'application
        Config              mConfig;
};

#endif // APPLICATION_HPP
