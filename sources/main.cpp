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

csvfile csv("image.csv");

int main(int argc, char *argv[])
{   

    //QApplication app(argc, argv);

    srand(time(0));

    try
    {
        // Construction du réseau de neurones
        unsigned int nombreInputGen = 1200;
        //Le Generateur
        std::vector<unsigned int> sizesGen{ {nombreInputGen,1000,800,784} };
        std::vector<Functions::ActivationFun> funsGen{ {Functions::sigmoid(0.1f), Functions::sigmoid(0.1f), Functions::sigmoid(0.1f)} };
        std::shared_ptr<NeuralNetwork> generator(new NeuralNetwork(sizesGen, funsGen));
        //Le Discriminateur
        std::vector<unsigned int> sizesDis{ {784,1000,300,1} };
        std::vector<Functions::ActivationFun> funsDis{ {Functions::sigmoid(0.1f), Functions::sigmoid(0.1f), Functions::sigmoid(0.1f)} };
        std::shared_ptr<NeuralNetwork> discriminator(new NeuralNetwork(sizesDis, funsDis));

        //Chargement de MNIST
        mnist_reader readerTrain("MNIST/train-images-60k", "MNIST/train-labels-60k");
        std::vector<Eigen::MatrixXf> imageTrain;
        Eigen::MatrixXi labelTrain;
        readerTrain.ReadMNIST(imageTrain, labelTrain);

        //Création du vecteur de bruit pour les tests
        std::vector<Eigen::MatrixXf> vectorTest;
        int sizeTest(20);
        for(int i(0); i < sizeTest; i++)
        {
            Eigen::MatrixXf noise = Eigen::MatrixXf::Random(nombreInputGen,1);
            vectorTest.push_back(noise);
        }

        Application::Batch batchTrain;
        Application::Batch batchTest;

        //Création du Batch de Test
        for(auto i(0); i< sizeTest; i++)
        {
            Eigen::MatrixXf outputTest = Eigen::MatrixXf::Zero(1,1);
            outputTest(0) = 0;
            batchTest.push_back(Application::Sample(vectorTest[i], outputTest));
        }
        cout << "Chargement du Batch de test effectué !" << endl;

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
        appMNIST.runExperiments(1, 100, 100);

        //Génération de 10 images et export pour analyse
        std::vector<Eigen::MatrixXf> resultat;
        for(int i(0); i < 10; i++)
        {
            Eigen::MatrixXf input = Eigen::MatrixXf::Random(nombreInputGen,1);
            resultat.push_back(appMNIST.genProcessing(input));
            for(int j(0); j<784; j++)
            {
                csv << resultat[i](j);
                if (j%28 == 27) csv << endrow;
            }
            csv << endrow;
        }
        cout << "Génération des images terminé !" << endl;

       /*for(int i(0); i < 5; i++)
        {
            Eigen::MatrixXf input = Eigen::MatrixXf::Random(1200,1);
            resultat.push_back(appMNIST.genProcessing(input));
            if (i==0)
            {
                QImage myImage(28,28, QImage::Format_Grayscale8);
                for(int j(0); j<784; j++)
                {
                    myImage.setPixel(j/28, j%28, resultat[i](j));
                }
                QLabel myLabel;
                myLabel.setPixmap(QPixmap::fromImage(myImage));

                myLabel.show();
            }
        }*/
    }
    catch (const std::exception& ex)
    {
        std::cout << "Exception was thrown: " << ex.what() << std::endl;
    }
    return 0;
}
