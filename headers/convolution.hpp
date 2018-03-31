#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <eigen3/Eigen/Dense>

class Convolution
{
public :
    Convolution(Eigen::MatrixXf input, Eigen::MatrixXf filtre, std::shared_ptr<Eigen::MatrixXf> resultat, bool sommerLignes);

    void operator()(int id);

    Eigen::MatrixXf mInput;
    Eigen::MatrixXf mFiltre;
    std::shared_ptr<Eigen::MatrixXf> mResultat;
    bool mSommerLignes;

};

#endif // CONVOLUTION_HPP
