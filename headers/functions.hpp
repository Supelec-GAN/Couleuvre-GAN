#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <functional>
#include <eigen3/Eigen/Dense>

/// Classe statique pour la gestion des fonctions
/**
 * Cette classe statique définit des alias de fonctions et permet d'en générer à la volée
 */
struct Functions
{
        /// Un alias sur une fonction d'activation
        using ActivationFun = std::function<float(float)>;
        /// Un alias sur une fonction d'erreur
        using ErrorFun      = std::function<float(Eigen::MatrixXf, Eigen::MatrixXf)>;

	
	
        /// Fonction d'activation sigmoide
        /**
         * @return f(x) = 1/(1-e(-x.lambda))
         * @param lambda le paramètre de pente de la sigmoide
         */
        static ActivationFun    sigmoid(float lambda);
        /// Fonction d'activation seuil
        /**
         * @return f(x) = 1 si x>= gapAbscissa, 0 sinon
         * @param gapAbscissa l'abscisse pour laquelle le seuil se fait
         */
        static ActivationFun    heavyside(float gapAbscissa);
        /// Fonction d'activation seuil
        /**
         * @return f(x) = 2/3 * tanh(1,7153*x)
         */
        static ActivationFun    hyperTan();

	
	
        ///Fonction d'erreur norme 2
        static ErrorFun         l2Norm();

        ///Fonction d'erreur pour le D du GAN
        static ErrorFun         disCout();

        static ErrorFun         genHeuristic();

        static ErrorFun         genKLDiv();

        static ErrorFun         genMinMax();
};

#endif // FUNCTIONS_HPP
