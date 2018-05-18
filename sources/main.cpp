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

#include "headers/application.hpp"
#include "headers/inputOutput/mnist_reader.h"
#include "headers/inputOutput/cifar10_reader.hpp"
#include "headers/inputOutput/cifar10provider.hpp"


using namespace std;
using namespace Eigen;

int main()
{
    /*MatrixXf inp = MatrixXf::Zero(1,16);
    for(int i(0); i < 16; i++)
    {
        inp(0,i) = 1;
    }
    std::vector<MatrixXf> vec;
    MatrixXf wei = MatrixXf::Zero(1,4);
    wei(0,2) =1;
    vec.push_back(wei);
    MatrixXf error = MatrixXf::Zero(1,9);
    error(0,8) = 1;

    for(int k(0); k < 1; k++)
    {
        cout << "k est : " << k << endl;
        if (k !=0) vec[0](0,k-1) = 0;
        //vec[0](0,0)=1;
        ConvolutionalLayer lay = ConvolutionalLayer(16,1,vec,Functions::reLu());
        for(int r(0); r<1;r++)
        {

        std::cout << "Printing Weight : ";
        for(int i(0); i < 4; i++)
        {
            if (i%2==0) cout << endl;
            cout << vec[0](r,i);
        }
        cout << endl;
        std::cout << "Printing Input : ";
        for(int i(0); i < 16; i++)
        {
            if (i%4==0) cout << endl;
            cout << inp(r,i);
            cout << " ";
        }
        cout << endl;
        std::cout << "Printing Process : ";
        for(int i(0); i < 9; i++)
        {
            if (i%3==0) cout << endl;
            cout << lay.processLayer(inp)(0,i);
            cout << " ";
        }
        cout << endl;
        MatrixXf res = lay.layerBackprop(error,1);
        std::cout << "Printing Error Backprop : ";
        for(int i(0); i<16; i++)
        {
            if (i%4==0) cout << endl;
            cout << res(0,i);
            cout << " ";
        }
        cout << endl << endl;
    }
    }*/

    //Construction de l'application qui gère tout
    Application appMNIST;
    appMNIST.runSingleExperimentMNIST();
    return 0;
}
