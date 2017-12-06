#ifndef NEURALNETWORK_INL
#define NEURALNETWORK_INL

template <typename Container>
NeuralNetwork::NeuralNetwork(Container layerList)
{
    for(auto itr = layerList.begin(); itr != layerList.end(); ++itr)
        push_back(*itr);
}

#endif // NEURALNETWORK_INL
