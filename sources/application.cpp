#include "headers/application.hpp"
#include <headers/rapidjson/error/en.h>

#include <math.h>
#include <fstream>
#include <ctime>

Application::Application(NeuralNetwork::Ptr discriminator, NeuralNetwork::Ptr generator, Batch teachingBatch, Batch testBatch)
: mDiscriminator(discriminator)
, mGenerator(generator)
, mTeacher(mGenerator, mDiscriminator)
, mTeachingBatch(teachingBatch)
, mTestingBatch(testBatch)
, mStatsCollector()
, mTestCounter(0)
{
    // Charge la configuration de l'application
    loadConfig();
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

#pragma mark - Expériences
//**************EXPERIENCES*************
//**************************************

void Application::runExperiments(unsigned int nbExperiments, unsigned int nbLoops, unsigned int nbTeachingsPerLoop)
{
    for(unsigned int index{0}; index < nbExperiments; ++index)
    {
        resetExperiment();
        runSingleStochasticExperiment(nbLoops, nbTeachingsPerLoop);
        std::cout << "Exp num. " << (index+1) << " finie !" << std::endl;
    }


    mStatsCollector.exportData(true);
}

void Application::runSingleStochasticExperiment(unsigned int nbLoops, unsigned int nbTeachingsPerLoop)
{
    mStatsCollector[0].addResult(runTest());
    bool trigger = false; //A changer si vous voulez faire des expériences funs
    for(unsigned int loopIndex{0}; loopIndex < nbLoops; ++loopIndex)
    {
        std::cout << "Apprentissage num. : " << (loopIndex)*nbTeachingsPerLoop << std::endl;
        runStochasticTeach(nbTeachingsPerLoop, trigger);
        auto score = runTest();
        mStatsCollector[loopIndex+1].addResult(score);
        std::cout << "Le score est de " << score << " et le trigger est en " << trigger << " !"<< std::endl;

        //Création Image
        if (loopIndex%100==0)
        {
            Eigen::MatrixXf input = Eigen::MatrixXf::Random(1, mGenerator->getInputSize());
            mStatsCollector.exportImage(mGenerator->processNetwork(input), loopIndex*nbTeachingsPerLoop);
        }
    }
}

void Application::resetExperiment()
{
    mGenerator->reset();
    mDiscriminator->reset();
}

#pragma mark - Apprentissage
//************APPRENTISSAGE*************
//**************************************

void Application::runStochasticTeach(unsigned int nbTeachings, bool trigger)
{
	
    std::uniform_int_distribution<> distribution(0, static_cast<int>(mTeachingBatch.size())-1);
    std::mt19937 randomEngine((std::random_device())());

    for(unsigned int index{0}; index < nbTeachings; index++)
    {
        Eigen::MatrixXf noiseInput = Eigen::MatrixXf::Random(1, mGenerator->getInputSize());
        Eigen::MatrixXf desiredOutput = Eigen::MatrixXf(1,1);
		Eigen::MatrixXf input = mGenerator->processNetwork(noiseInput);

        desiredOutput(0,0) = 1;
        mTeacher.backpropGenerator(input, desiredOutput, mConfig.step, mConfig.dx);

        if (trigger)
        {
            noiseInput = Eigen::MatrixXf::Random(1, mGenerator->getInputSize());
            input = mGenerator->processNetwork(noiseInput);

            desiredOutput(0,0) = 1;
            mTeacher.backpropGenerator(input, desiredOutput, mConfig.step, mConfig.dx);
        }
        else
        {
            Sample sample{mTeachingBatch[distribution(randomEngine)]};
            mTeacher.backpropDiscriminator(sample.first, sample.second, mConfig.step, mConfig.dx);

            desiredOutput(0,0) = 0;
            mTeacher.backpropDiscriminator(input, desiredOutput, mConfig.step, mConfig.dx);
        }
    }
}

void Application::runMiniBatchTeach(unsigned int nbTeachings, unsigned int batchSize)
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
        for(std::vector<Sample>::iterator itr = mTestingBatch.begin(); itr != mTestingBatch.end() && limit-- != 0; ++itr)
        {
            Eigen::MatrixXf output{mDiscriminator->processNetwork(mGenerator->processNetwork(itr->first))};
            errorMean += sqrt((output - itr->second).squaredNorm());
        }
    }
    return errorMean/static_cast<float>(mTestingBatch.size());
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

#pragma mark - Configuration
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

    *mStatsCollector.getCSVFile() << "Step" << mConfig.step << "dx" << mConfig.dx << endrow;
}

void Application::exportPoids() //A Implementer
{
    //Generateur
    csvfile csvGen("generateur.csv");

    //Discriminateur
    csvfile csvDis("discriminateur.csv");

}
