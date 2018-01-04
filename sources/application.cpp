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

double minimumScoreBeforeTrigger = 0.1 ; 

void Application::runSingleStochasticExperiment(unsigned int nbLoops, unsigned int nbTeachingsPerLoop)
{
    mStatsCollector[0].addResult(runTest());
    bool trigger = false;
    for(unsigned int loopIndex{0}; loopIndex < nbLoops; ++loopIndex)
    {
        std::cout << "Apprentissage num. : " << (loopIndex)*nbTeachingsPerLoop << std::endl;
        runStochasticTeach(nbTeachingsPerLoop, trigger);
        auto score = runTest();
        mStatsCollector[loopIndex+1].addResult(score);
        if (score < minimumScoreBeforeTrigger) trigger = true;
        else trigger = false;
        std::cout << "Le score est de " << score << " et le trigger est en " << trigger << " !"<< std::endl;
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
	//generate a random distribution (to later randomly select elements from the mTeachingBatch)
    std::uniform_int_distribution<> distribution(0, static_cast<int>(mTeachingBatch.size())-1);
    std::mt19937 randomEngine((std::random_device())());
	
    for(unsigned int index{0}; index < nbTeachings; index++)
    {
		//Teach the Generator
		//generate the noise used for the input of the generator, then create the generated image
        Eigen::MatrixXf noiseInput = Eigen::MatrixXf::Random(1, mGenerator->getInputSize());
		Eigen::MatrixXf input = mGenerator->processNetwork(noiseInput);

		//set the desired output for the discriminator as a true image
		Eigen::MatrixXf desiredOutput = Eigen::MatrixXf(1,1);
        desiredOutput(0,0) = 1;
		
		//Apply Backprop to Gen
        mTeacher.backpropGenerator(input, desiredOutput, mConfig.step, mConfig.dx);

        if (trigger) //then teach the Generator a second time
        {
			
            noiseInput = Eigen::MatrixXf::Random(1, mGenerator->getInputSize());
            input = mGenerator->processNetwork(noiseInput);

            desiredOutput(0,0) = 1; //true image
            mTeacher.backpropGenerator(input, desiredOutput, mConfig.step, mConfig.dx);
        }
        else //teach the Discriminator
			//(this way of teaching is not correct as per the algorithm described in Goodfellow et al. 2014
			//backpropagation must be done simultaneously
        {
			//teach the Discriminator on a true image randomly selected in the mTeachingBatch
            Sample sample{mTeachingBatch[distribution(randomEngine)]};
            mTeacher.backpropDiscriminator(sample.first, sample.second, mConfig.step, mConfig.dx);

			//teach the Discriminator on the previously generated image
            desiredOutput(0,0) = 0; //generated image
            mTeacher.backpropDiscriminator(input, desiredOutput, mConfig.step, mConfig.dx);
        }
    }
}

//==============================================================================================================

void Application::runMinibatchTeach(unsigned int nbTeachings, unsigned int minibatchSize)
//Algorithm 1 p.4 of generative-adversarial-nets by Goodfellow et al. 2014
//Minibatch stockastic gradient descent training of generative adversarial nets
{
	for(unsigned int index{0}; index < nbTeachings; index++)
	{
#warning finish implementing

		Eigen::MatrixXf desiredOutput = Eigen::MatrixXf(1,1);
		
		for (unsigned long k(0); k < 1; k++)
		{
			//"Sample minibatch of batchSize noise samples {z_1, ..., z_m} from noise prior p_g(z)"
			Minibatch generatedImagesFromNoiseMinibatch = sampleGeneratedImagesFromNoiseMinibatch(minibatchSize);
			
			//"Sample minibatch of batchSize examples {x_1, ..., x_m} from data-generating distribution p_data(x)
			Minibatch exampleMinibatch = sampleMinibatch(mTeachingBatch,minibatchSize);
			
			//"Update the discriminator by ascending its stochastic gradient"
			
			//(this way of teaching is not correct as per the algorithm described in Goodfellow et al. 2014
			//backpropagation must be done simultaneously
			
			//teach the Discriminator on a true image randomly selected in the mTeachingBatch
//			Sample sample{mTeachingBatch[distribution(randomEngine)]};
//			mTeacher.backpropDiscriminator(sample.first, sample.second, mConfig.step, mConfig.dx);
			
			//teach the Discriminator on the previously generated image
//			desiredOutput(0,0) = 0; //generated image
//			mTeacher.backpropDiscriminator(input, desiredOutput, mConfig.step, mConfig.dx);
		}
			 
		//Teach the Generator
		
		//"Sample minibatch of batchSize noise samples {z_1, ..., z_m} from noise prior p_g(z)"
		Minibatch generatedImagesFromNoiseMinibatch = sampleGeneratedImagesFromNoiseMinibatch(minibatchSize);
		
		//"Update the generator by descending the stochastic gradient"
		//Apply Backprop to Gen
//		mTeacher.backpropGenerator(input, desiredOutput, mConfig.step, mConfig.dx);
		
		

	}
}

//============================================================================================================



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

Application::Minibatch Application::sampleMinibatch(Application::Batch batch, unsigned long minibatchSize)
{
	Application::Minibatch minibatch(minibatchSize);
	
	//Tirage aléatoire sans remise
	std::vector<unsigned long> randomizedIntVector(batch.size());
	std::iota(randomizedIntVector.begin(), randomizedIntVector.end(), 0); 		//fills in with first int numbers starting at 0
	std::random_shuffle(randomizedIntVector.begin(),randomizedIntVector.end());
	
	for (unsigned long i(0); i < minibatchSize ; ++i)
	{
		minibatch[i] = batch[randomizedIntVector[i]];
	}
	return minibatch;
}

Application::Minibatch Application::sampleGeneratedImagesFromNoiseMinibatch(unsigned long minibatchSize)
{
	Application::Minibatch generatedImagesFromNoiseMinibatch(minibatchSize);
	
	Eigen::MatrixXf desiredOutput = Eigen::MatrixXf(1,1);
	desiredOutput(0,0) = 1; //generator want to create near-real images

	for (unsigned long i(0); i < minibatchSize ; ++i)
	{
		Eigen::MatrixXf noiseInput = Eigen::MatrixXf::Random(1, mGenerator->getInputSize());
		Eigen::MatrixXf input = mGenerator->processNetwork(noiseInput);
		
		Sample imageSample = std::make_pair(input, desiredOutput);
		
		generatedImagesFromNoiseMinibatch[i] = imageSample;
	}
	return generatedImagesFromNoiseMinibatch;
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
