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

        //Création du Batch d'entrainement du discriminateur
        for(auto i(0); i< mConfig.labelTrainSize; i++)
        {
            Eigen::MatrixXf outputTrain = Eigen::MatrixXf::Zero(1,1);
            outputTrain(0,0) = 1;
            if (labelTrain(i) == mConfig.chiffreATracer)
            {
                mTeachingBatchDis.push_back(Application::Sample(imageTrain[i], outputTrain));
            }
        }
        cout << "Chargement du Batch d'entrainement effectué !" << endl;

        //Création du Batch de test du discriminateur
        for(auto i(0); i<mConfig.labelTestSize; i++)
			
        {
            Eigen::MatrixXf outputTest = Eigen::MatrixXf::Zero(1,1);
            outputTest(0) = 1;
			if (labelTest(i) == mConfig.chiffreATracer)
            {
                mTestingBatchDis.push_back(Application::Sample(imageTest[i], outputTest));
            }
        }
		
		//Création du vecteur de bruit pour les tests du générateur
		std::vector<Eigen::MatrixXf> vectorTest;
		for(int i(0); i < mConfig.nbGenTest; i++)
		{
			Eigen::MatrixXf noise = Eigen::MatrixXf::Random(1, mConfig.genLayerSizes[0] );
			vectorTest.push_back(noise);
		}
		
		//Création du Batch de Test du générateur
		for(auto i(0); i< mConfig.nbGenTest; i++)
		{
			Eigen::MatrixXf outputTest = Eigen::MatrixXf::Zero(1,1);
			outputTest(0) = 0;
			mTestingBatchGen.push_back(Application::Sample(vectorTest[i], outputTest));
		}
		std::cout << "Chargement du Batch de test effectué !" << std::endl;
		
		
		if (mConfig.networkAreImported)
		{
			mDiscriminator = NeuralNetwork::Ptr(importNeuralNetwork(mConfig.discriminatorPath,Functions::sigmoid(mConfig.sigmoidParameter)));
			std::cout << "Chargement du Discriminateur effectué !" << std::endl;
			
			mGenerator = NeuralNetwork::Ptr(importNeuralNetwork(mConfig.generatorPath,Functions::sigmoid(mConfig.sigmoidParameter)));
			std::cout << "Chargement du Générateur effectué !" << std::endl;
		}
		else
		{
			// Construction du réseau de neurones
			//Le Generateur
			std::vector<Functions::ActivationFun> funsGen;
			for(int i(0); i < mConfig.genLayerSizes.size()-1;i++)
				funsGen.push_back(Functions::sigmoid(mConfig.sigmoidParameter));
			mGenerator = NeuralNetwork::Ptr(new NeuralNetwork(mConfig.genLayerSizes, funsGen));
			//Le Discriminateur
			std::vector<Functions::ActivationFun> funsDis;
			
			for(int i(0); i < mConfig.disLayerSizes.size()-1;i++)
				funsDis.push_back(Functions::sigmoid(mConfig.sigmoidParameter));
			mDiscriminator = NeuralNetwork::Ptr(new NeuralNetwork(mConfig.disLayerSizes , funsDis));
		}
		mTeacher = Teacher(mGenerator,mDiscriminator);
		mTestCounter = 0;
		
    }
    catch (const std::exception& ex)
    {
        std::cout << "Exception was thrown: " << ex.what() << std::endl;
    }
}

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
		
		if (mConfig.typeOfExperiment == "Stochastic")
		{
			runSingleStochasticExperiment();
		}
		else if (mConfig.typeOfExperiment == "Minibatch")
		{
			runSingleMinibatchExperiment();
		}
		else
		{
			std::cout << "Application::runExperiments error : typeOfExperiment is unknown (" << stderr << ")" << std::endl;
			exit(EXIT_FAILURE);
		}
		exportPoids();
        std::cout << "Exp num. " << (index+1) << " finie !" << std::endl;
    }

    mStatsCollector.exportData(true);
}

void Application::runSingleStochasticExperiment()
{
    mStatsCollector[0].addResultGen(runTestGen());
    bool trigger = false; //A changer si vous voulez faire des expériences funs
    for(unsigned int loopIndex{0}; loopIndex < mConfig.nbLoopsPerExperiment; ++loopIndex)
    {
        std::cout << "Apprentissage num. : " << (loopIndex)*mConfig.nbTeachingsPerLoop << std::endl;
        runStochasticTeach(trigger);
        auto scoreGen = runTestGen();
        auto scoreDis = runTestDis(mConfig.nbDisTest);
        mStatsCollector[loopIndex+1].addResultGen(scoreGen);
		mStatsCollector[loopIndex+1].addResultDis(scoreDis);
		std::cout << "Le scoreGen est de " << scoreGen << " et le scoreDis de " << scoreDis << " !" << std::endl;
		//Création Image
		if (loopIndex%mConfig.intervalleImg==0)
		{
			Eigen::MatrixXf input = Eigen::MatrixXf::Random(1, mGenerator->getInputSize());
			mStatsCollector.exportImage(mGenerator->processNetwork(input), loopIndex*mConfig.nbTeachingsPerLoop);
		}
		
	}
}

void Application::runSingleMinibatchExperiment()
{
	mStatsCollector[0].addResultGen(runTestGen());
	for(unsigned int loopIndex{0}; loopIndex < mConfig.nbLoopsPerExperiment; ++loopIndex)
	{
		std::cout << "Apprentissage num. : " << (loopIndex)*mConfig.nbTeachingsPerLoop << std::endl;
		runMinibatchTeach();
		auto scoreGen = runTestGen();
		auto scoreDis = runTestDis(mConfig.nbDisTest);
		mStatsCollector[loopIndex+1].addResultGen(scoreGen);
		mStatsCollector[loopIndex+1].addResultDis(scoreDis);
		std::cout << "Le scoreGen est de " << scoreGen << " et le scoreDis de " << scoreDis << " !" << std::endl;
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
	std::uniform_int_distribution<> distribution(0, static_cast<int>(mTeachingBatchDis.size())-1);
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
			Sample sample{mTeachingBatchDis[distribution(randomEngine)]};
			mTeacher.backpropDiscriminator(sample.first, sample.second, mConfig.step, mConfig.dx);
			
			Eigen::MatrixXf input = mGenerator->processNetwork(noiseInput);
			desiredOutput(0,0) = 0;
			mTeacher.backpropDiscriminator(input, desiredOutput, mConfig.step, mConfig.dx);
		}
	}
}

void Application::runMinibatchTeach()
{
	unsigned int minibatchWeightingCoefficient = mConfig.useAverageForBatchlearning ? mConfig.minibatchSize : 1;
	
	for(unsigned int index{0}; index < mConfig.nbTeachingsPerLoop; index++)
	{
		Eigen::MatrixXf desiredOutput0 = Eigen::MatrixXf(1,1);
		Eigen::MatrixXf desiredOutput1 = Eigen::MatrixXf(1,1);
		desiredOutput0(0,0) = 0;
		desiredOutput1(0,0) = 1;
		for (unsigned long k(0); k < 1; ++k)
		{
			Minibatch generatedImagesFromNoiseMinibatch = sampleGeneratedImagesFromNoiseMinibatch(); 	//"Sample minibatch of batchSize noise samples {z_1, ..., z_m} from noise prior p_g(z)"
			Minibatch exampleMinibatch = sampleMinibatch(mTeachingBatchDis);							//"Sample minibatch of batchSize examples {x_1, ..., x_m} from data-generating distribution p_data(x)

			//"Update the discriminator by ascending its stochastic gradient"
			for (unsigned long i(0); i < mConfig.minibatchSize; ++i)
			{
				Sample falseimagesample{generatedImagesFromNoiseMinibatch[i]};
				Sample trueimagesample{exampleMinibatch[i]};
				mTeacher.minibatchDiscriminatorBackprop(mDiscriminator,falseimagesample.first, desiredOutput0, mConfig.step, mConfig.dx);
				mTeacher.minibatchDiscriminatorBackprop(mDiscriminator,trueimagesample.first, desiredOutput1, mConfig.step, mConfig.dx);
			}
			mTeacher.updateNetworkWeights(mDiscriminator, minibatchWeightingCoefficient);
		}
		Minibatch generatedImagesFromNoiseMinibatch = sampleGeneratedImagesFromNoiseMinibatch(); 		//"Sample minibatch of batchSize noise samples {z_1, ..., z_m} from noise prior p_g(z)"

		for(std::vector<Sample>::iterator itr = generatedImagesFromNoiseMinibatch.begin(); itr != generatedImagesFromNoiseMinibatch.end(); ++itr) 	//"Update the generator by descending the stochastic gradient"
		{
			Sample sample{*itr};
			mTeacher.minibatchGeneratorBackprop(mGenerator,sample.first, desiredOutput1, mConfig.step, mConfig.dx);
		}
		mTeacher.updateNetworkWeights(mGenerator, minibatchWeightingCoefficient);
	}
}

float Application::runTestGen(int limit, bool returnErrorRate)
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

[[deprecated]]
float Application::gameScore(int nbImages)
{
	float mean = 0;
	for (int i(0); i < nbImages; i++)
	{
		mean += (mDiscriminator->processNetwork(mGenerator->processNetwork(Eigen::MatrixXf::Random(1, mGenerator->getInputSize()))))(0);
	}
	return(mean/(float)nbImages);
}

[[deprecated]]
Eigen::MatrixXf Application::genProcessing(Eigen::MatrixXf input)
{
	return(mGenerator->processNetwork(input));
}

Application::Minibatch Application::sampleMinibatch(Application::Batch batch)
{
	Application::Minibatch minibatch(mConfig.minibatchSize);
	
	//Tirage aléatoire sans remise
	std::vector<unsigned long> randomizedIntVector(batch.size());
	std::iota(randomizedIntVector.begin(), randomizedIntVector.end(), 0); 		//fills in with first int numbers starting at 0
	std::random_shuffle(randomizedIntVector.begin(),randomizedIntVector.end());
	
	for (unsigned long i(0); i < mConfig.minibatchSize ; ++i)
	{
		minibatch[i] = batch[randomizedIntVector[i]];
	}
	return minibatch;
}

Application::Minibatch Application::sampleGeneratedImagesFromNoiseMinibatch()
{
	Application::Minibatch generatedImagesFromNoiseMinibatch(mConfig.minibatchSize);
	
	
	Eigen::MatrixXf desiredOutput = Eigen::MatrixXf(1,1);
	desiredOutput(0,0) = 1;

	for (unsigned long i(0); i < mConfig.minibatchSize ; ++i)
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
	mConfig.sigmoidParameter = document["sigmoidParameter"].GetFloat();


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
	mConfig.nbDisTest = document["nbDisTest"].GetUint();
	mConfig.nbGenTest = document["nbGenTest"].GetUint();
	mConfig.labelTrainSize = document["labelTrainSize"].GetUint();
	mConfig.labelTestSize = document["labelTestSize"].GetUint();
    mConfig.intervalleImg = document["intervalleImg"].GetUint();
    mConfig.chiffreATracer = document["chiffreATracer"].GetUint();
	mConfig.minibatchSize = document["minibatchSize"].GetUint();


    mConfig.generatorPath = document["generatorPath"].GetString();
    mConfig.discriminatorPath = document["discriminatorPath"].GetString();

    mConfig.generatorDest = document["generatorDest"].GetString();
    mConfig.discriminatorDest = document["discriminatorDest"].GetString();
	mConfig.CSVFileNameResult = document["CSVFileNameResult"].GetString();
	mConfig.CSVFileNameImage = document["CSVFileNameImage"].GetString();
	
	mConfig.typeOfExperiment = document["typeOfExperiment"].GetString();
	mConfig.useAverageForBatchlearning = document["useAverageForBatchlearning"].GetBool();

	

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
