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

int main()
{

    //QApplication app(argc, argv);

	srand(static_cast<unsigned int>(time(0)) );


    //Construction de l'application qui gère tout
    Application appCifar;
    appCifar.runExperiments();

    return 0;
}
