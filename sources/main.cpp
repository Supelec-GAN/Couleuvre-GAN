#include <iostream>
#include <random>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <ctime>

#include "headers/application.hpp"
#include "headers/mnist_reader.h"

using namespace std;

int main()
{   
    srand(time(0));

    try
    {
        // Construction du réseau de neurones
        std::vector<unsigned int> sizes{ {784,1000,300,10} };
        std::vector<Functions::ActivationFun> funs{ {Functions::sigmoid(0.1f), Functions::sigmoid(0.1f), Functions::sigmoid(0.1f)} };
        std::shared_ptr<NeuralNetwork> network(new NeuralNetwork(sizes, funs));

        mnist_reader readerTrain("MNIST/train-images-60k", "MNIST/train-labels-60k");
        std::vector<Eigen::MatrixXf> imageTrain;
        Eigen::MatrixXi labelTrain;
        readerTrain.ReadMNIST(imageTrain, labelTrain);

        mnist_reader readerTest("MNIST/test-images-10k", "MNIST/test-labels-10k");
        std::vector<Eigen::MatrixXf> imageTest;
        Eigen::MatrixXi labelTest;
        readerTest.ReadMNIST(imageTest, labelTest);

        Application::Batch batchTrain;
        Application::Batch batchTest;

        for(auto i(0); i< labelTrain.size(); i++)
        {
            Eigen::MatrixXf outputTrain = Eigen::MatrixXf::Zero(10,1);
            outputTrain(labelTrain(i)) = 1;
            batchTrain.push_back(Application::Sample(imageTrain[i], outputTrain));
        }
        cout << "Chargement du Batch d'entrainement effectué !" << endl;
        for(auto i(0); i< labelTest.size(); i++)
        {
            Eigen::MatrixXf outputTest = Eigen::MatrixXf::Zero(10,1);
            outputTest(labelTest(i)) = 1;
            batchTest.push_back(Application::Sample(imageTest[i], outputTest));
        }
        cout << "Chargement du Batch de test effectué !" << endl;

        //Construction de l'application qui gère tout
        Application appMNIST(network, batchTrain, batchTest);
        appMNIST.runExperiments(1, 20, 60000);
    }
    catch (const std::exception& ex)
    {
        std::cout << "Exception was thrown: " << ex.what() << std::endl;
    }
    return 0;
}
