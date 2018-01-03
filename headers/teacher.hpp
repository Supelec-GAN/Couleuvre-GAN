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
         *  @param generator un smart pointeur sur le générateur dont on veut superviser l'apprentissage
		 *  @param discriminator un smart pointeur sur le discriminateur dont on veut superviser l'apprentissage
         */
                        Teacher(NeuralNetwork::Ptr generator, NeuralNetwork::Ptr discriminator);

        /// Constructeur par pointer
        /**
         *  Construit un teacher supervisant l'apprentissage d'un réseau de neurone
		 *  @param generator un pointeur sur le générateur dont on veut superviser l'apprentissage
		 *  @param discriminator un pointeur sur le discriminateur dont on veut superviser l'apprentissage
         */
                        Teacher(NeuralNetwork* generator, NeuralNetwork* discriminator);


#pragma mark - Backpropagation
	
        /// Fonction appliquant la méthode de rétropropagation sur mDiscriminator
        /**
         * Calcule la première dérivée dE/dXn puis propage l'erreur à travers le réseau
         * @param input le vecteur d'input que le réseau va process
         * @param desiredOutput la sortie modèle dont on veut se rapprocher
         * @param step le pas d'apprentissage
         * @param dx le deplacement élémentaire pour calculer la dérivée
         */
        void            backpropDiscriminator(Eigen::MatrixXf input, Eigen::MatrixXf desiredOutput, float step = 0.2, float dx = 0.05);

        /// Fonction appliquant la méthode de rétropropagation sur mGenerator
        /**
         * Calcule la première dérivée dE/dXn puis propage l'erreur à travers le réseau
         * @param input le vecteur d'input que le réseau va process
         * @param desiredOutput la sortie modèle dont on veut se rapprocher
         * @param step le pas d'apprentissage
         * @param dx le deplacement élémentaire pour calculer la dérivée
         */

        void            backpropGenerator(Eigen::MatrixXf input, Eigen::MatrixXf desiredOutput, float step = 0.2, float dx = 0.05);

		/// Fonction appliquant la méthode de rétropropagation par mini-Batch sur mNetwork
		/**
		 * Calcule la première dérivée dE/dXn puis propage l'erreur à travers le réseau
		 * @param input le vecteur d'input que le réseau va process
		 * @param desiredOutput la sortie modèle dont on veut se rapprocher
		 * @param step le pas d'apprentissage
		 * @param dx le deplacement élémentaire pour calculer la dérivée
		 */
		void 			miniBatchBackProp(Eigen::VectorXf input,Eigen::VectorXf desiredOutput, float step = 0.2, float dx = 0.05);
	
		/// Fonction mettant à jour les poids du réseau
		void 			updateNetworkWeights();
	
    private:
        /// Fonction propageant l'erreur itérativement à travers le réseau discriminant
        /**
         * La fonction itère sur toutes les couches de neurones et appliques les formules de récurrence
         * @param xnPartialDerivative la dérivée dE/dXn initiale
         * @param step le pas d'apprentissage
         */
        void            propagateErrorDiscriminator(Eigen::MatrixXf xnPartialDerivative, float step);



        /// Fonction propageant l'erreur itérativement à travers le réseau discriminant
        /**
         * La fonction itère sur toutes les couches de neurones et propage l'erreur sans faire varier les poids.
         * @param xnPartialDerivative la dérivée dE/dXn initiale
         */
        Eigen::MatrixXf propagateErrorDiscriminatorInvariant(Eigen::MatrixXf xnPartialDerivative);

        /// Fonction propageant l'erreur itérativement à travers le réseau générateur
        /**
         * La fonction itère sur toutes les couches de neurones et appliques les formules de récurrence
         * @param xnPartialDerivative la dérivée dE/dXn initiale
         * @param step le pas d'apprentissage
         */
        void            propagateErrorGenerator(Eigen::MatrixXf xnPartialDerivative, float step);

        /// Fonction calculant le vecteur dE/dXn initial
        /**
         * La fonction effectue la dérivée de la fonction d'erreur par rapport à une variation dans chaque
         * dimension, successivement
         * @param output la sortie obtenue
         * @param desiredOutput la sortie modèle
         * @param dx le pas de dérivation
         * @return renvoie le vecteur dE/dXn
         */
        Eigen::MatrixXf calculateInitialErrorVector(Eigen::MatrixXf output, Eigen::MatrixXf desiredOutput, float dx);

        /// Fonction calculant le vecteur dE/dXn initial
        /**
         * La fonction effectue la dérivée de la fonction d'erreur par rapport à une variation dans chaque
         * dimension, successivement
         * @param output la sortie obtenue du réseau générateur
         * @param desiredOutput la sortie modèle après passage par le disciminateur
         * @param dx le pas de dérivation
         * @return renvoie le vecteur dE/dXn
         */
        Eigen::MatrixXf calculateInitialErrorVectorGen(Eigen::MatrixXf output, Eigen::MatrixXf desiredOutput, float dx);

    private:
        /// Des pointeur sur les réseaux dont on veut superviser l'apprentissage
        NeuralNetwork::Ptr  mGenerator;
        NeuralNetwork::Ptr  mDiscriminator;

        /// La fonction d'erreur utilisée
        Functions::ErrorFun mErrorFun;
};

#endif // TEACHER_HPP
