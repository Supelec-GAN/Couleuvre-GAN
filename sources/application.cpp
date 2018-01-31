#include "headers/application.hpp"
#include <headers/rapidjson/error/en.h>

#include <math.h>
#include <fstream>
#include <ctime>
#include <iostream>

Application::Application() :
mStatsCollector()
{
    // Charge la configuration de l'application
    loadConfig();

    try
    {
        //Chargement de MNIST
        mnist_reader readerTrain("MNIST/train-images-60k", "MNIST/train-labels-60k");
        std::vector<Eigen::MatrixXf> imageTrain;
        Eigen::MatrixXi labelTrain;
        readerTrain.ReadMNIST(imageTrain, labelTrain);

        mnist_reader readerTest("MNIST/test-images-10k", "MNIST/test-labels-10k");
        std::vector<Eigen::MatrixXf> imageTest;
        Eigen::MatrixXi labelTest;
        readerTest.ReadMNIST(imageTest, labelTest);

        //Création du Batch d'entrainement
        for(auto i(0); i< labelTrain.size(); i++)
        {
            Eigen::MatrixXf outputTrain = Eigen::MatrixXf::Zero(1,1);
            outputTrain(0,0) = 1;
            if (labelTrain(i) == mConfig.chiffreATracer)
            {
                mTeachingBatch.push_back(Application::Sample(imageTrain[i], outputTrain));
            }
        }
        cout << "Chargement du Batch d'entrainement effectué !" << endl;

        //Création du Batch de test
        for(auto i(0); i<1000 /*labelTest.size()*/; i++)
        {
            Eigen::MatrixXf outputTest = Eigen::MatrixXf::Zero(1,1);
            outputTest(0) = 1;
            if (labelTest(i) = mConfig.chiffreATracer)
            {
                mTestingBatchDis.push_back(Application::Sample(imageTest[i], outputTest));
            }
        }
    }
    catch (const std::exception& ex)
    {
        std::cout << "Exception was thrown: " << ex.what() << std::endl;
    }

    //Création du vecteur de bruit pour les tests
    std::vector<Eigen::MatrixXf> vectorTest;
    int sizeTest(20);
    for(int i(0); i < sizeTest; i++)
    {
        Eigen::MatrixXf noise = Eigen::MatrixXf::Random(1, mConfig.genLayerSizes[0] );
        vectorTest.push_back(noise);
    }

    //Création du Batch de Test
    for(auto i(0); i< sizeTest; i++)
    {
        Eigen::MatrixXf outputTest = Eigen::MatrixXf::Zero(1,1);
        outputTest(0) = 0;
        mTestingBatchGen.push_back(Application::Sample(vectorTest[i], outputTest));
    }
    std::cout << "Chargement du Batch de test effectué !" << std::endl;


    if (mConfig.networkAreImported)
    {
        mDiscriminator = NeuralNetwork::Ptr(importNeuralNetwork(mConfig.discriminatorPath,Functions::sigmoid(0.1f)));
        std::cout << "Chargement du Discriminateur effectué !" << std::endl;

        mGenerator = NeuralNetwork::Ptr(importNeuralNetwork(mConfig.generatorPath,Functions::sigmoid(0.1f)));
        std::cout << "Chargement du Générateur effectué !" << std::endl;
    }
    else
    {
        // Construction du réseau de neurones
        //Le Generateur
        std::vector<Functions::ActivationFun> funsGen;
        for(int i(0); i < mConfig.genLayerSizes.size()-1;i++)
            funsGen.push_back(Functions::sigmoid(0.1f));
        mGenerator = NeuralNetwork::Ptr(new NeuralNetwork(mConfig.genLayerSizes, funsGen));
        //Le Discriminateur
        std::vector<Functions::ActivationFun> funsDis;

        for(int i(0); i < mConfig.disLayerSizes.size()-1;i++)
            funsDis.push_back(Functions::sigmoid(0.1f));
        mDiscriminator = NeuralNetwork::Ptr(new NeuralNetwork(mConfig.disLayerSizes , funsDis));
    }
    mTeacher = Teacher(mGenerator,mDiscriminator);
    mTestCounter = 0;
}

/*Application::Application(   NeuralNetwork::Ptr network,
                            std::function<Eigen::MatrixXf (Eigen::MatrixXf)> modelFunction,
                            std::vector<Eigen::MatrixXf> teachingInputs,
                            std::vector<Eigen::MatrixXf> testingInputs)
: mNetwork(network)
, mTeacher(mNetwork)
, mStatsCollector()
, mTestCounter(0)
{
    // Charge la configuration de l'application
    loadConfig();

    // Génère le batch d'apprentissage à partir des entrées et de la fonction à modéliser
    for(size_t i{0}; i < teachingInputs.size(); ++i)
        mTeachingBatch.push_back(Sample(teachingInputs[i], modelFunction(teachingInputs[i])));
    // Génère le batch d'apprentissage à partir des entrées et de la fonction à modéliser
    for(size_t i{0}; i < testingInputs.size(); ++i)
        mTestingBatch.push_back(Sample(testingInputs[i], modelFunction(testingInputs[i])));
}*/

////#pragma mark - Expériences
//**************EXPERIENCES*************
//**************************************

void Application::runExperiments()
{
    for(unsigned int index{0}; index < mConfig.nbExperiments; ++index)
    {
        if (!mConfig.networkAreImported)
        {
            resetExperiment();
            std::cout << "Réseau réinitialisé !" << std::endl;
        }
        runSingleStochasticExperiment();
        exportPoids();
        std::cout << "Exp num. " << (index+1) << " finie !" << std::endl;
    }

    mStatsCollector.exportData(true);
}

void Application::runSingleStochasticExperiment()
{
    mStatsCollector[0].addResult(runTest());
    bool trigger = false; //A changer si vous voulez faire des expériences funs
    for(unsigned int loopIndex{0}; loopIndex < mConfig.nbLoopsPerExperiment; ++loopIndex)
    {
        std::cout << "Apprentissage num. : " << (loopIndex)*mConfig.nbTeachingsPerLoop << std::endl;
        runStochasticTeach(trigger);
        auto score = runTest();
        auto scoreDis = runTestDis();
        mStatsCollector[loopIndex+1].addResult(score);
        mStatsCollector[loopIndex+1].addResultDis(scoreDis);
        std::cout << "Le score est de " << score << " et le scoreDis de " << scoreDis << " !" << std::endl;
        //Création Image
        if (loopIndex%mConfig.intervalleImg==0)
        {
            Eigen::MatrixXf input = Eigen::MatrixXf::Random(1, mGenerator->getInputSize());
            mStatsCollector.exportImage(mGenerator->processNetwork(input), loopIndex*mConfig.nbTeachingsPerLoop);
        }
    }
}

void Application::resetExperiment()
{
    mGenerator->reset();
    mDiscriminator->reset();
}

////#pragma mark - Apprentissage
//************APPRENTISSAGE*************
//**************************************

void Application::runStochasticTeach(bool trigger)
{
    std::uniform_int_distribution<> distribution(0, static_cast<int>(mTeachingBatch.size())-1);
    std::mt19937 randomEngine((std::random_device())());

    for(unsigned int index{0}; index < mConfig.nbTeachingsPerLoop; index++)
    {
        Eigen::MatrixXf noiseInput = Eigen::MatrixXf::Random(1, mGenerator->getInputSize());
        Eigen::MatrixXf desiredOutput = Eigen::MatrixXf(1,1);

        for(int i(0); i<mConfig.nbGenTeach; i++)
        {

            Eigen::MatrixXf noiseInput = Eigen::MatrixXf::Random(1, mGenerator->getInputSize());
            Eigen::MatrixXf input = mGenerator->processNetwork(noiseInput);
            noiseInput = Eigen::MatrixXf::Random(1, mGenerator->getInputSize());
            input = mGenerator->processNetwork(noiseInput);

            desiredOutput(0,0) = 1;
            mTeacher.backpropGenerator(input, desiredOutput, mConfig.step, mConfig.dx);
        }
        for(int i(0); i<mConfig.nbDisTeach; i++)
        {
            noiseInput = Eigen::MatrixXf::Random(1, mGenerator->getInputSize());
            Sample sample{mTeachingBatch[distribution(randomEngine)]};
            mTeacher.backpropDiscriminator(sample.first, sample.second, mConfig.step, mConfig.dx);

            Eigen::MatrixXf input = mGenerator->processNetwork(noiseInput);
            desiredOutput(0,0) = 0;
            mTeacher.backpropDiscriminator(input, desiredOutput, mConfig.step, mConfig.dx);
        }
    }
}

void Application::runMiniBatchTeach(unsigned int batchSize)
{
#warning Japillow must implement
	throw std::logic_error("Not implemented yet");
//	for (unsigned int i(0); i < nbTeachings; ++i){
//		auto samples(generateBatch(batchSize));
//		for(auto itr = samples.begin(); itr != samples.end(); ++itr)
//			mTeacher.miniBatchBackProp(itr->first, itr->second);
//		mTeacher.updateNetworkWeights();
//	}
}

float Application::runTest(int limit, bool returnErrorRate)
{
    float errorMean{0};
    if (returnErrorRate)
    {
        for(std::vector<Sample>::iterator itr = mTestingBatchGen.begin(); itr != mTestingBatchGen.end() && limit-- != 0; ++itr)
        {
            Eigen::MatrixXf output{mDiscriminator->processNetwork(mGenerator->processNetwork(itr->first))};
            errorMean += sqrt((output - itr->second).squaredNorm());
        }
    }
    return errorMean/static_cast<float>(mTestingBatchGen.size());
}


float Application::runTestDis(int limit, bool returnErrorRate)
{
    float errorMean{0};
    if (returnErrorRate)
    {
        for(std::vector<Sample>::iterator itr = mTestingBatchDis.begin(); itr != mTestingBatchDis.end() && limit-- != 0; ++itr)
        {
            Eigen::MatrixXf output{mDiscriminator->processNetwork(itr->first)};
            errorMean += sqrt((output).squaredNorm());
        }
    }
    return errorMean/static_cast<float>(mTestingBatchDis.size());
}

float Application::gameScore(int nbImages)
{
	float mean = 0;
	for (int i(0); i < nbImages; i++)
	{
		mean += (mDiscriminator->processNetwork(mGenerator->processNetwork(Eigen::MatrixXf::Random(1, mGenerator->getInputSize()))))(0);
	}
	return(mean/(float)nbImages);
}

Eigen::MatrixXf Application::genProcessing(Eigen::MatrixXf input)
{
	return(mGenerator->processNetwork(input));
}

////#pragma mark - Configuration
//************CONFIGURATION*************
//**************************************

void Application::loadConfig(const std::string& configFileName)
{
    std::stringstream ss;
    std::ifstream inputStream(configFileName);
    if(!inputStream)
    {
      throw std::runtime_error("Application::loadConfig Error - Failed to load " + configFileName);
    }
    ss << inputStream.rdbuf();
    inputStream.close();
    rapidjson::Document doc;
    rapidjson::ParseResult ok(doc.Parse(ss.str().c_str()));
    if(!ok)
    {
        std::cout << stderr << "JSON parse error: %s (%u)" << rapidjson::GetParseError_En(ok.Code()) << ok.Offset() << std::endl;
        exit(EXIT_FAILURE);
    }

    setConfig(doc);
}

void Application::setConfig(rapidjson::Document& document)
{
    mConfig.step = document["step"].GetFloat();
    mConfig.dx = document["dx"].GetFloat();

    mConfig.networkAreImported = document["networkAreImported"].GetBool();

    auto layersSizesDis = document["layersSizesDis"].GetArray();
    for(rapidjson::SizeType i = 0; i < layersSizesDis.Size(); i++)
        mConfig.disLayerSizes.push_back(layersSizesDis[i].GetUint());

    auto layersSizesGen = document["layersSizesGen"].GetArray();
    for(rapidjson::SizeType i = 0; i < layersSizesGen.Size(); i++)
        mConfig.genLayerSizes.push_back(layersSizesGen[i].GetUint());

    mConfig.nbExperiments = document["nbExperiments"].GetUint();
    mConfig.nbLoopsPerExperiment = document["nbLoopsPerExperiment"].GetUint();
    mConfig.nbTeachingsPerLoop = document["nbTeachingsPerLoop"].GetUint();
    mConfig.nbDisTeach = document["nbDisTeach"].GetUint();
    mConfig.nbGenTeach = document["nbGenTeach"].GetUint();
    mConfig.intervalleImg = document["intervalleImg"].GetUint();
    mConfig.chiffreATracer = document["chiffreATracer"].GetUint();

    mConfig.generatorPath = document["generatorPath"].GetString();
    mConfig.discriminatorPath = document["discriminatorPath"].GetString();

    mConfig.generatorDest = document["generatorDest"].GetString();
    mConfig.discriminatorDest = document["discriminatorDest"].GetString();

    *mStatsCollector.getCSVFile() << "Step" << mConfig.step << "dx" << mConfig.dx << endrow;
}

void Application::exportPoids()
{
    csvfile csvGen(mConfig.generatorDest);
    for(unsigned int i(0); i < mConfig.genLayerSizes.size(); i++)
       csvGen << mConfig.genLayerSizes[i];
    csvGen << endrow;
    csvGen << *mGenerator;

    csvfile csvDis(mConfig.discriminatorDest);
    for(unsigned int i(0); i < mConfig.disLayerSizes.size(); i++)
       csvDis << mConfig.disLayerSizes[i];
    csvDis << endrow;
    csvDis << *mDiscriminator;
    std::cout << "Export des réseaux effectués !" << std::endl;
}

NeuralNetwork* Application::importNeuralNetwork(std::string networkPath,Functions::ActivationFun activationFun)
{
    std::ifstream ifs (networkPath);
    std::string a;
    std::vector<Eigen::MatrixXf> neuralNetwork;
    std::vector<Eigen::MatrixXf> bias;
    std::vector<unsigned int> taille;
    int k = 0;
    getline(ifs, a,'\n');
    std::string b = "";
    for(auto i(0); i < a.length(); i++)
    {
        if (a[i] == ';')
        {
            if (b != "")

            {
                taille.push_back(stoi(b));
                b = "";
            }
        }
        else
            b = b + a[i];
    }
    for(auto i(0); i < taille.size()-1; i++)
    {
        neuralNetwork.push_back(Eigen::MatrixXf::Zero(taille[i],taille[i+1]));
        bias.push_back(Eigen::MatrixXf::Zero(1,taille[i+1]));
    }
    std::vector<Functions::ActivationFun> activationFunVector;
    for(auto i(0); i < taille.size()-1; i++)
    {
        activationFunVector.push_back(activationFun);
    }
    int i = 0;
    int j = 0;
    while(k < taille.size()-1)
    {
        std::getline(ifs, a,'\n');
        if (a != "")
        {
            for(auto itr=a.begin(); itr != a.end(); itr++)
            {
                if (*itr == ';')
                {
                    if (b != "")
                    {
                        neuralNetwork[k](j,i) = (std::stof(b));
                        i++;
                        b = "";
                    }
                }
                else
                    b = b + *itr;
            }
        j++;
        i = 0;
        }
        else
        {
            j = 0;
            std::getline(ifs, a,'\n');
            for(auto itr=a.begin(); itr != a.end(); itr++)
            {
                if (*itr == ';')
                {
                    if (b != "")
                    {
                        bias[k](0,j) = (std::stof(b));
                        j++;
                        b = "";
                    }
                }
                else
                    b = b + *itr;
            }
            j=0;
            std::getline(ifs, a, '\n');
            std::getline(ifs, a, '\n');
            k = k+1;
        }
    }
    ifs.close();
    return (new NeuralNetwork(taille, neuralNetwork, bias, activationFunVector));
}
