#ifndef MNIST_READER_H
#define MNIST_READER_H
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <string>
#include <vector>

using namespace std;

class mnist_reader
{
public:
    mnist_reader(string full_path_image, string full_path_label);
    void ReadMNIST(vector<Eigen::MatrixXf> &mnist, Eigen::MatrixXi &label);

private:
    static int reverseInt (int i);
    string mFullPathImage;
    string mFullPathLabel;
};

#endif // MNIST_READER_H
