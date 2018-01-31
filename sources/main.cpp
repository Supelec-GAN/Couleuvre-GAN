#include <iostream>
#include <random>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <ctime>
#include <string>
/*#include <QImage>
#include <QApplication>
#include <QWidget>
#include <QLabel>*/

#include "headers/CSVFile.h"
#include "headers/application.hpp"
#include "headers/mnist_reader.h"

using namespace std;

int main(int argc, char *argv[])
{       
    csvfile csv("imagefinale.csv");

    //QApplication app(argc, argv);

	srand(static_cast<unsigned int>(time(0)) );

    try
    {
        //Chargement de MNIST
        mnist_reader readerTrain("MNIST/train-images-60k", "MNIST/train-labels-60k");
        std::vector<Eigen::MatrixXf> imageTrain;
        Eigen::MatrixXi labelTrain;
        readerTrain.ReadMNIST(imageTrain, labelTrain);

        Application::Batch batchTrain;

        //Création du Batch d'entrainement
        for(auto i(0); i< labelTrain.size(); i++)
        {
            Eigen::MatrixXf outputTrain = Eigen::MatrixXf::Zero(1,1);
            outputTrain(0,0) = 1;
            if (true)
            {
                batchTrain.push_back(Application::Sample(imageTrain[i], outputTrain));
            }
        }
        cout << "Chargement du Batch d'entrainement effectué !" << endl;

        //Construction de l'application qui gère tout
        Application appMNIST(batchTrain);
        appMNIST.runExperiments();

        //Génération de 10 images et export pour analyse
/*        std::vector<Eigen::MatrixXf> resultat;
        for(int i(0); i < 10; i++)
        {
            Eigen::MatrixXf input = Eigen::MatrixXf::Random(1, nombreInputGen);
            resultat.push_back(appMNIST.genProcessing(input));
            for(int j(0); j<784; j++)
            {
                csv << resultat[i](j);
                if (j%28 == 27) csv << endrow;
            }
            csv << endrow;
        }
        cout << "Génération des images terminé !" << endl;
*/

    }
    catch (const std::exception& ex)
    {
        std::cout << "Exception was thrown: " << ex.what() << std::endl;
    }
    return 0;
}
