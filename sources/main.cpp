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

    //QApplication app(argc, argv);

	srand(static_cast<unsigned int>(time(0)) );


    //Construction de l'application qui gère tout
    Application appMNIST;
    appMNIST.runExperiments();




        //Création du Batch d'entrainement
        for(auto i(0); i< labelTrain.size(); i++)
        {
            Eigen::MatrixXf outputTrain = Eigen::MatrixXf::Zero(1,1);
            outputTrain(0,0) = 1;
            if (labelTrain(i) == 1)
            {
                batchTrain.push_back(Application::Sample(imageTrain[i], outputTrain));
            }
        }
        cout << "Chargement du Batch d'entrainement effectué !" << endl;

        //Construction de l'application qui gère tout
        Application appMNIST(discriminator, generator, batchTrain, batchTest);
		appMNIST.runExperiments(1, 100, 100,"Minibatch", 1);

        //Génération de 10 images et export pour analyse
        std::vector<Eigen::MatrixXf> resultat;
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

    }
    catch (const std::exception& ex)
    {
        std::cout << "Exception was thrown: " << ex.what() << std::endl;
    }
    return 0;
}
