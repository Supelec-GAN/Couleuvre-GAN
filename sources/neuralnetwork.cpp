#include "headers/neuralnetwork.hpp"

//#pragma mark - Constructeur
//*************CONSTRUCTEUR*************
//**************************************

NeuralNetwork::NeuralNetwork(){}

NeuralNetwork::NeuralNetwork(std::vector<unsigned int> layerSizes, std::vector<Functions::ActivationFun> activationFuns)
{
    if(layerSizes.size() != activationFuns.size() + 1)
        throw std::logic_error("NeuralNetwork::NeuralNetwork error - Sizes of parameters do not match");

    for(size_t i(0); i < layerSizes.size()-1; ++i)
        push_back(NeuronLayer(layerSizes[i], layerSizes[i+1], activationFuns[i]));
}

NeuralNetwork::NeuralNetwork(std::vector<unsigned int> layerSizes)
{
    for(size_t i(0); i < layerSizes.size()-1; ++i)
        push_back(NeuronLayer(layerSizes[i], layerSizes[i+1]));
}

NeuralNetwork::NeuralNetwork(std::vector<unsigned int> layerSizes, std::vector<Eigen::MatrixXf> weightVector, std::vector<Eigen::MatrixXf> biasVector, std::vector<Functions::ActivationFun> activationFuns)
{
    if(layerSizes.size() != activationFuns.size() + 1)
        throw std::logic_error("NeuralNetwork::NeuralNetwork error - Sizes of parameters do not match");

    for(size_t i(0); i < layerSizes.size()-1; ++i)
        push_back(NeuronLayer(layerSizes[i], layerSizes[i+1], weightVector[i], biasVector[i], activationFuns[i]));
}

//#pragma mark - Propagation
//*************PROPAGATION**************
//**************************************

Eigen::MatrixXf NeuralNetwork::processNetwork(Eigen::MatrixXf input)
{
	for(auto itr = begin(); itr != end(); ++itr)
		input = (*itr).processLayer(input);
	
	return input;
}

Eigen::MatrixXf NeuralNetwork::processNetwork()
{
    Eigen::MatrixXf input = Eigen::MatrixXf::Random((*begin()).getInputSize(), 1);
	for(auto itr = begin(); itr != end(); ++itr)
        input = (*itr).processLayer(input);
	
	return input;
}

//#pragma mark - Autres
//****************AUTRES****************
//**************************************

void NeuralNetwork::reset()
{
    for (auto itr = begin(); itr != end(); ++itr)
        itr->reset();
}

int NeuralNetwork::getInputSize()
{
    return ((*(begin())).getInputSize());
}


//#pragma mark - Auxiliaires
//*************AUXILIAIRES**************
//**************************************

std::ostream& operator<<(std::ostream& flux, NeuralNetwork network)
{
    NeuralNetwork::iterator it;
    for (it = network.begin(); it != network.end(); it++)
    {
        flux << *it << "\n" << std::endl;
    }
    return flux;
}

