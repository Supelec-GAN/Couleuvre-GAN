#include "headers/application.hpp"
#include <headers/rapidjson/error/en.h>

#include <math.h>
#include <fstream>

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

void Application::runExperiments(unsigned int nbExperiments, unsigned int nbLoops, unsigned int nbTeachingsPerLoop)
{
    for(unsigned int index{0}; index < nbExperiments; ++index)
    {
        runSingleExperiment(nbLoops, nbTeachingsPerLoop);
        std::cout << "Exp num. " << (index+1) << " finie !" << std::endl;
        resetExperiment();
    }

    mStatsCollector.exportData(true);
}

void Application::runSingleExperiment(unsigned int nbLoops, unsigned int nbTeachingsPerLoop)
{
    mStatsCollector[0].addResult(runTest());

    for(unsigned int loopIndex{0}; loopIndex < nbLoops; ++loopIndex)
    {
        std::cout << "Apprentissage num. : " << (loopIndex)*nbTeachingsPerLoop << std::endl;
        runTeach(nbTeachingsPerLoop);
        mStatsCollector[loopIndex+1].addResult(runTest());
    }
}

void Application::resetExperiment()
{
    mGenerator->reset();
    mDiscriminator->reset();
}

void Application::runTeach(unsigned int nbTeachings)
{
    std::uniform_int_distribution<> distribution(0, mTeachingBatch.size()-1);
    std::mt19937 randomEngine((std::random_device())());

    for(unsigned int index{0}; index < nbTeachings; index++)
    {
        Eigen::MatrixXf noiseInput = Eigen::MatrixXf::Random(mGenerator->getInputSize(),1);
        Eigen::MatrixXf desiredOutput = Eigen::MatrixXf(1,1);
        desiredOutput(0,0) = 1;
        Eigen::MatrixXf input = mGenerator->process(noiseInput);
        mTeacher.backPropGen(input, desiredOutput, mConfig.step, mConfig.dx);

        if (index%2 == 0)
        {
            Sample sample{mTeachingBatch[distribution(randomEngine)]};
            mTeacher.backPropDis(sample.first, sample.second, mConfig.step, mConfig.dx);
        }
        else
        {
            Eigen::MatrixXf desiredOutput = Eigen::MatrixXf(1,1);
            desiredOutput(0,0) = 0;
            mTeacher.backPropDis(input, desiredOutput, mConfig.step, mConfig.dx);
        }
        if(index %100 == 0)
            std::cout << "+" << index << std::endl;
    }
}

float Application::runTest(int limit, bool returnErrorRate)
{
    float errorMean{0};
    if (returnErrorRate)
    {
        for(std::vector<Sample>::iterator itr = mTestingBatch.begin(); itr != mTestingBatch.end() && limit-- != 0; ++itr)
        {
            Eigen::MatrixXf output{mDiscriminator->process(mGenerator->process(itr->first))};
            errorMean += sqrt((output - itr->second).squaredNorm());
        }
    }
    return errorMean/static_cast<float>(mTestingBatch.size());
}

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

Eigen::MatrixXf Application::genProcessing(Eigen::MatrixXf input)
{
    return(mGenerator->process(input));
}

