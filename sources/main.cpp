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
    /*MatrixXf inp = MatrixXf::Zero(2,16);
    for(int i(0); i < 16; i++)
    {
        inp(0,i) = 2*i;
        inp(1,i) = 2*i+1;
    }
    std::vector<MatrixXf> vec;
    MatrixXf wei = MatrixXf::Zero(2,4);
    vec.push_back(wei);
    for(int k(0); k < 1; k++)
    {
        cout << "k est : " << k << endl;
        if (k !=0) vec[0](0,k-1) = 0;
        vec[0](0,0)=1;
        vec[0](1,0)=1;
        ConvolutionalLayer lay = ConvolutionalLayer(16,2,vec,Functions::reLu());
        for(int r(0); r<2;r++)
        {
        for(int i(0); i < 4; i++)
        {
            if (i%2==0) cout << endl;
            cout << vec[0](r,i);
        }
        cout << endl;
        for(int i(0); i < 16; i++)
        {
            if (i%4==0) cout << endl;
            cout << inp(r,i);
            cout << " ";
        }
        cout << endl;
        for(int i(0); i < 9; i++)
        {
            if (i%3==0) cout << endl;
            cout << lay.processLayer(inp)(0,i);
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
