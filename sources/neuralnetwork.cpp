#include "headers/neuralnetwork.hpp"

//*************CONSTRUCTEUR*************
//**************************************

NeuralNetwork::NeuralNetwork(){}

NeuralNetwork::NeuralNetwork(std::vector<unsigned int> layerTypes, std::vector<unsigned int> layerSizes, std::vector<unsigned int> layerChannels, std::vector<std::vector<unsigned int>> layerArg, std::vector<Functions::ActivationFun> activationFuns, unsigned int descentType)
{
    if(layerSizes.size() != activationFuns.size() + 1)
        throw std::logic_error("NeuralNetwork::NeuralNetwork error - Sizes of parameters do not match");

    for(size_t i(0); i < layerSizes.size()-1; ++i)
    {
        switch(layerTypes[i]) {
            case 0 : push_back(NeuronLayer::Ptr(new FullConnectedLayer(layerSizes[i], layerSizes[i+1], activationFuns[i])));
                break;
            case 1 :
                break;
            case 2 : push_back(NeuronLayer::Ptr(new ConvolutionalLayer(layerSizes[i], layerChannels[i], sqrt(layerSizes[i])-sqrt(layerSizes[i+1])+1, layerChannels[i+1])));
                break;
            case 3 : push_back(NeuronLayer::Ptr(new NoisyLayer(layerSizes[i], layerSizes[i+1], activationFuns[i])));
                break;
            case 4 : push_back(NeuronLayer::Ptr(new ZeroPadLayer(layerSizes[i],layerSizes[i+1],layerArg[i][0])));
                break;
        }
    }
}

/*NeuralNetwork::NeuralNetwork(std::vector<unsigned int> layerSizes)
{
    for(size_t i(0); i < layerSizes.size()-1; ++i)
        push_back(NeuronLayer(layerSizes[i], layerSizes[i+1]));
}

NeuralNetwork::NeuralNetwork(std::vector<unsigned int> layerSizes, std::vector<Eigen::MatrixXf> weightVector, std::vector<Eigen::MatrixXf> biasVector, std::vector<Functions::ActivationFun> activationFuns, unsigned int descentType)
{
    if(layerSizes.size() != activationFuns.size() + 1)
        throw std::logic_error("NeuralNetwork::NeuralNetwork error - Sizes of parameters do not match");

    for(size_t i(0); i < layerSizes.size()-1; ++i)
        push_back(NeuronLayer(layerSizes[i], layerSizes[i+1], weightVector[i], biasVector[i], activationFuns[i], descentType));
}*/

//*************PROPAGATION**************
//**************************************

Eigen::MatrixXf NeuralNetwork::processNetwork(Eigen::MatrixXf input)
{
	for(auto itr = begin(); itr != end(); ++itr)
       input = (*itr)->processLayer(input);
	
	return input;
}

Eigen::MatrixXf NeuralNetwork::processNetwork()
{
    Eigen::MatrixXf input = Eigen::MatrixXf::Random((*begin())->getInputSize(), 1);
	for(auto itr = begin(); itr != end(); ++itr)
        input = (*itr)->processLayer(input);
	
	return input;
}

//****************AUTRES****************
//**************************************

void NeuralNetwork::reset()
{
    for (auto itr = begin(); itr != end(); ++itr)
        (*itr)->reset();
}

int NeuralNetwork::getInputSize()
{
    return (*(begin()))->getInputSize();
}


//*************AUXILIAIRES**************
//**************************************

std::ostream& operator<<(std::ostream& flux, NeuralNetwork network)
{
    /*NeuralNetwork::iterator it;
    for (it = network.begin(); it != network.end(); it++)
    {
        flux << *it << "\n" << std::endl;
    }*/
    return flux;
}

