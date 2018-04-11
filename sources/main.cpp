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

int main()
{

    //QApplication app(argc, argv);

	srand(static_cast<unsigned int>(time(0)) );


    //Construction de l'application qui gère tout
    Application appMNIST;
    appMNIST.runExperiments();

    return 0;
}
