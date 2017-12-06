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
         * @param network le réseau avec lequel on travaille
         * @param teachingBatch le batch des données servant à l'apprentissage
         * @param testingBatch le batch des données de test
         */
        Application(NeuralNetwork::Ptr network, Batch teachingBatch, Batch testingBatch);

        /// Constructeur par fonction modèle
        /**
         * Ce constructeur supervise le projet par rapport au réseaude neurones donné, des batchs d'inputs pour l'apprentissage et les tests,
         * et la fonction à modéliser
         * @param network le réseau avec lequel on travaille
         * @param modelFunction la fonction à modéliser
         * @param teachingInputs les inputs pour l'apprentissage
         * @param testingInputs les inputs pour les tests
         */
        Application(NeuralNetwork::Ptr network,
                    std::function<Eigen::MatrixXf(Eigen::MatrixXf)> modelFunction,
                    std::vector<Eigen::MatrixXf> teachingInputs,
                    std::vector<Eigen::MatrixXf> testingInputs);

        /// Effectue une run d'apprentissage
        /**
         * Effectue une run d'apprentissage dont le nombre d'apprentissages est passé en paramètres
         * @param nbTeachings le nombre d'apprentissages à faire pendant la run
         */
        void runTeach(unsigned int nbTeachings);

        /// Effectue une run de tests
        /**
         * Effectue une run de test dont le nombre de tests est passé en paramètres
         * @param nbTests le nombre de tests à faire pendant la run
         */
        float runTest(int limit = -1, bool returnErrorRate = 1);

        void runExperiments(unsigned int nbExperiments, unsigned int nbLoops, unsigned int nbTeachingsPerLoop);
        void runSingleExperiment(unsigned int nbLoops, unsigned int nbTeachingsPerLoop);

        void resetExperiment();

    private:
        /// Fonction pour charger la configuration de l'application
        void loadConfig(const std::string& configFileName = "config.json");
        void setConfig(rapidjson::Document& document);

    private:
        /// Les réseaux avec lequel on travaille
        NeuralNetwork::Ptr  mDiscrimator;
        NeuralNetwork::Ptr  mGenerator;
        /// Le teacher qui permet de superviser l'apprentissage du réseau
        Teacher             mTeacherDis;
        Teacher             mTeacherGen;

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
