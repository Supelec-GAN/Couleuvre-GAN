#ifndef TEACHER_HPP
#define TEACHER_HPP

#include "headers/neuralnetwork.hpp"
#include "headers/functions.hpp"

class Teacher
{
    public:

        /// Constructeur par unique pointer
        /**
         *  Construit un teacher supervisant l'apprentissage d'un réseau de neurone
         *  @param network un smart pointeur sur le réseau dont on veut superviser l'apprentissage
         */
                        Teacher(NeuralNetwork::Ptr generator, NeuralNetwork::Ptr discriminator);

        /// Constructeur par pointer
        /**
         * Construit un teacher supervisant l'apprentissage d'un réseau de neurone
         * @param network un pointeur sur le réseau dont on veut superviser l'apprentissage
         */
                        Teacher(NeuralNetwork* generator, NeuralNetwork* discriminator);



        /// Fonction appliquant la méthode de rétropropagation sur mNetwork
        /**
         * Calcule la première dérivée dE/dXn puis propage l'erreur à travers le réseau
         * @param input le vecteur d'input que le réseau va process
         * @param desiredOutput la sortie modèle dont on veut se rapprocher
         * @param step le pas d'apprentissage
         * @param dx le deplacement élémentaire pour calculer la dérivée
         */
        void            backPropDis(Eigen::MatrixXf input, Eigen::MatrixXf desiredOutput, float step = 0.2, float dx = 0.05);

        void            backPropGen(Eigen::MatrixXf input, Eigen::MatrixXf desiredOutput, float step = 0.2, float dx = 0.05);

    private:
        /// Fonction propageant l'erreur itérativement à travers le réseau
        /**
         * La fonction itère sur toutes les couches de neurones et appliques les formules de récurrence
         * @param xnPartialDerivative la dérivée dE/dXn initiale
         * @param step le pas d'apprentissage
         */
        void            propErrorDis(Eigen::MatrixXf xnPartialDerivative, float step);

        void            propErrorGen(Eigen::MatrixXf xnPartialDerivative, float step);

        /// Fonction calculant le vecteur dE/dXn initial
        /**
         * La fonction effectue la dérivée de la fonction d'erreur par rapport à une variation dans chaque
         * dimension, successivement
         * @param output la sortie obtenue
         * @param desiredOutput la sortie modèle
         * @param dx le pas de dérivation
         * @return renvoie le vecteur dE/dXn
         */
        Eigen::MatrixXf errorVector(Eigen::MatrixXf output, Eigen::MatrixXf desiredOutput, float dx);

        Eigen::MatrixXf errorVectorGen(Eigen::MatrixXf output, Eigen::MatrixXf desiredOutput, float dx);

    private:
        /// Un pointeur sur le réseau dont on veut superviser l'apprentissage
        NeuralNetwork::Ptr  mGenerator;
        NeuralNetwork::Ptr  mDiscriminator;

        /// La fonction d'erreur utilisée
        Functions::ErrorFun mErrorFun;
};

#endif // TEACHER_HPP
