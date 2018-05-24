#ifndef CIFAR10PROVIDER_HPP
#define CIFAR10PROVIDER_HPP

#include "headers/inputOutput/inputprovider.hpp"

#include <type_traits>

class Cifar10Provider : public InputProvider
{
    public:
        // N'importe quel type d'entier peut remplacer uint16_t, mais il faut qu'il soit sur au moins 9 bits
        enum class CifarLabel : uint16_t
        {
            airplane = 1 << 0, // = 1
            automobile = 1 << 1, // = 2
            bird = 1 << 2, // = 4
            cat = 1 << 3, // = 8
            deer = 1 << 4, // = 16
            dog = 1 << 5, // = 32
            frog = 1 << 6, // = 64
            horse = 1 << 7, // = 128
            ship = 1 << 8, // = 256
            truck = 1 << 9 // = 512
        };

        using Utype = std::underlying_type<Cifar10Provider::CifarLabel>::type;

    public:
        /// Constructeur
        /**
         * @param labels une combinaison des 10 labels listés dans l'enum CifarLabel, qui correspondent aux classes avec lesquelles on veut travailler
         * @param labelTrainSize le nombre d'éléments du set de train que l'on veut utiliser
         * @param labelTestSize le nombre d'éléments du set de test que l'on veut utiliser
         */
        Cifar10Provider(CifarLabel labels, unsigned int labelTrainSize = 50000, unsigned int labelTestSize = 10000);

        /// Retourne le batch de training
        /**
        * @param greyLevel : détermine si l'on travaille sur les images en couleur ou en niveau de gris
        */
        Batch trainingBatch(bool greyLevel) const;
        /// Retourne le batch de test
        /**
        * @param greyLevel : détermine si l'on travaille sur les images en couleur ou en niveau de gris
        */
        Batch testingBatch(bool greyLevel) const;

    private:
        /// Permet de faire la conversion label (airplaine, dog...) vers un id cifar entre 0 et 9
        /**
         * @param label une combinaison des 10 labels listés dans l'enum CifarLabel
         * @param id un label numérique entre 0 et 9 associé aux images sur lesquelles on veut travailler
         * @return vrai si id correspond bien à une classe d'image qu'on veut traiter
         */
        static bool matchLabelWithId(CifarLabel label, uint8_t id);

        /// Retourne la ième matrice de training ou de test
        /**
         * @param index l'indice de l'image à recuperer dans le set
         * @param isTrainOrTestRequired true si on veut une image de training et false pour une de test
         * @param greyLevel : détermine si l'on travaille sur les images en couleur ou en niveau de gris
         * @return la matrice correspondant à l'image d'indice index dans le set spécifié
         */
        Eigen::MatrixXf getMatrix(unsigned int index, bool isTrainOrTestRequired = 1, bool greyLevel = 0) const;

    private:
        // Résolution automatique de type parce que j'ai la flemme
        decltype(cifar::read_dataset()) mDataset;
        CifarLabel                      mLabels;
};


inline Cifar10Provider::CifarLabel operator|(Cifar10Provider::CifarLabel lhs, Cifar10Provider::CifarLabel rhs)
{
    return static_cast<Cifar10Provider::CifarLabel>(static_cast<Cifar10Provider::Utype>(lhs) | static_cast<Cifar10Provider::Utype>(rhs));
}

inline Cifar10Provider::CifarLabel operator&(Cifar10Provider::CifarLabel lhs, Cifar10Provider::CifarLabel rhs)
{
    return static_cast<Cifar10Provider::CifarLabel>(static_cast<Cifar10Provider::Utype>(lhs) & static_cast<Cifar10Provider::Utype>(rhs));
}

std::ostream& operator<<(std::ostream& flux, Cifar10Provider::CifarLabel label);



#endif // CIFAR10PROVIDER_HPP
