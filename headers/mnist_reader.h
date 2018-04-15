#ifndef MNIST_READER_H
#define MNIST_READER_H
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <string>
#include <vector>

class mnist_reader
{
public:
    mnist_reader(std::string full_path_image, std::string full_path_label);
    void ReadMNIST(std::vector<Eigen::MatrixXf> &mnist, Eigen::MatrixXi &label);

private:
    static int reverseInt (int i);
    std::string mFullPathImage;
    std::string mFullPathLabel;
};

#endif // MNIST_READER_H
