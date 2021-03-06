#include "headers/application.hpp"
#include "headers/inputOutput//cifar10provider.hpp"
#include <headers/rapidjson/error/en.h>

#include <math.h>
#include <fstream>
#include <ctime>
#include <iostream>

Application::Application()
{
    // Charge la configuration de l'application
    loadConfig();

    /// A réparer, cette fonctionnalité est pétée (et n'a jamais marché en fait)
    /// mStatsCollector : Stats::StatsCollector(mConfig.CSVFileNameResult,mConfig.CSVFileNameImage);
    *(mStatsCollector.getCSVFile()) << "Step" << mConfig.step << "dx" << mConfig.dx << endrow;

    try
    {
        if (mConfig.databaseToUse == "mnist")
        {
            InputProvider::Ptr inputProvider(new MnistProvider(mConfig.chiffresATracer, 6000, 1000));
            mTeachingBatchDis = inputProvider->trainingBatch();
            mTestingBatchDis = inputProvider->testingBatch();
        }
        else if (mConfig.databaseToUse == "cifar10")
        {
            //Cifar10Provider::CifarLabel CifVehicle =    Cifar10Provider::CifarLabel::airplane | Cifar10Provider::CifarLabel::automobile | Cifar10Provider::CifarLabel::ship | Cifar10Provider::CifarLabel::truck;
            Cifar10Provider::CifarLabel CifAnimals =   Cifar10Provider::CifarLabel::bird | Cifar10Provider::CifarLabel::cat |
Cifar10Provider::CifarLabel::deer | Cifar10Provider::CifarLabel::dog | Cifar10Provider::CifarLabel::horse | Cifar10Provider::CifarLabel::frog;
            //Cifar10Provider::CifarLabel CifAll = CifAnimals | CifVehicle;
            InputProvider::Ptr inputProvider(new Cifar10Provider(CifAnimals, 10000, 10000));
            mTeachingBatchDis = inputProvider->trainingBatch(1);
            mTestingBatchDis = inputProvider->testingBatch(1);
        }
        else
        {
            std::cout << "Application::Application error : databaseToUse is unknown (" << stderr << ")" << std::endl;
            exit(EXIT_FAILURE);
        }

		//Création du vecteur de bruit pour les tests du générateur
		std::vector<Eigen::MatrixXf> vectorTest;
        for(unsigned int i(0); i < mConfig.nbGenTest; i++)
		{
            Eigen::MatrixXf noise = Eigen::MatrixXf::Random(mConfig.genLayerNbChannels[0],mConfig.genLayerSizes[0] );
			vectorTest.push_back(noise);
		}
		
		//Création du Batch de Test du générateur
        for(unsigned int i(0); i< mConfig.nbGenTest; i++)
		{
			Eigen::MatrixXf outputTest = Eigen::MatrixXf::Zero(1,1);
			outputTest(0) = 0;
			mTestingBatchGen.push_back(Application::Sample(vectorTest[i], outputTest));
		}
		std::cout << "Chargement du Batch de test effectué !" << std::endl;
		
		
		if (mConfig.networkAreImported)
		{
            /*mDiscriminator = NeuralNetwork::Ptr(importNeuralNetwork(mConfig.discriminatorPath,Functions::sigmoid(mConfig.sigmoidParameter)));
			std::cout << "Chargement du Discriminateur effectué !" << std::endl;

			mGenerator = NeuralNetwork::Ptr(importNeuralNetwork(mConfig.generatorPath,Functions::sigmoid(mConfig.sigmoidParameter)));
            std::cout << "Chargement du Générateur effectué !" << std::endl;*/
		}
		else
		{
			// Construction du réseau de neurones
			//Le Generateur
			std::vector<Functions::ActivationFun> funsGen;
			for(int i(0); i < mConfig.genLayerSizes.size()-1;i++)
            {
                if (i==mConfig.genLayerSizes.size()-2) funsGen.push_back(Functions::sigmoid(0.1f));
                else funsGen.push_back(Functions::sigmoid(0.1f));
            }
            mGenerator = NeuralNetwork::Ptr(new NeuralNetwork(mConfig.genLayerTypes, mConfig.genLayerSizes, mConfig.genLayerNbChannels, mConfig.genLayerArgs, funsGen, mConfig.descentTypeGen));
			//Le Discriminateur
			std::vector<Functions::ActivationFun> funsDis;
			for(int i(0); i < mConfig.disLayerSizes.size()-1;i++)
                funsDis.push_back(Functions::sigmoid(0.1f));
            mDiscriminator = NeuralNetwork::Ptr(new NeuralNetwork(mConfig.disLayerTypes, mConfig.disLayerSizes, mConfig.disLayerNbChannels, mConfig.disLayerArgs, funsDis, mConfig.descentTypeDis));
		}
        mTeacher = Teacher(mGenerator,mDiscriminator, mConfig.genFunction);
		mTestCounter = 0;
		
    }
    catch (const std::exception& ex)
    {
        std::cout << "Exception was thrown: " << ex.what() << std::endl;
    }
}

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
		
		if (mConfig.typeOfExperiment == "stochastic")
		{
			runSingleStochasticExperiment();
		}
		else if (mConfig.typeOfExperiment == "minibatch")
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

    for(unsigned int loopIndex{0}; loopIndex < mConfig.nbLoopsPerExperiment; ++loopIndex)
    {
        std::cout << "Apprentissage num. : " << (loopIndex)*mConfig.nbTeachingsPerLoop << std::endl;
        runStochasticTeach();
        auto scoreGen = runTestGen();
        auto scoreDis = runTestDis(mConfig.nbDisTest);

        mStatsCollector[loopIndex+1].addResultGen(scoreGen);
		mStatsCollector[loopIndex+1].addResultDis(scoreDis);
		std::cout << "Le scoreGen est de " << scoreGen << " et le scoreDis de " << scoreDis << " !" << std::endl;
		//Création Image
		if (loopIndex%mConfig.intervalleImg==0)
		{
            Eigen::MatrixXf input;
            for(unsigned int i(0); i < mConfig.nbImgParIntervalleImg; i++)
            {
                input = Eigen::MatrixXf::Random(mConfig.genLayerNbChannels[0],mConfig.genLayerSizes[0]);
                mStatsCollector.exportImage(mGenerator->processNetwork(input), loopIndex*mConfig.nbTeachingsPerLoop, mConfig.imageSizeSide);
            }
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
            Eigen::MatrixXf input;
            for(unsigned int i(0); i < mConfig.nbImgParIntervalleImg; i++)
            {
                input = Eigen::MatrixXf::Random(mConfig.genLayerNbChannels[0],mConfig.genLayerSizes[0]);
                mStatsCollector.exportImage(mGenerator->processNetwork(input), loopIndex*mConfig.nbTeachingsPerLoop, mConfig.imageSizeSide);
            }
        }
	}
}

void Application::resetExperiment()
{
    mGenerator->reset();
    mDiscriminator->reset();
}

//************APPRENTISSAGE*************
//**************************************

void Application::runStochasticTeach()
{
	std::uniform_int_distribution<> distribution(0, static_cast<int>(mTeachingBatchDis.size())-1);
	std::mt19937 randomEngine((std::random_device())());
	
	for(unsigned int index{0}; index < mConfig.nbTeachingsPerLoop; index++)
	{
        Eigen::MatrixXf noiseInput = Eigen::MatrixXf::Random(mConfig.genLayerNbChannels[0],mConfig.genLayerSizes[0]);
		Eigen::MatrixXf desiredOutput = Eigen::MatrixXf(1,1);
		
        for(unsigned int i(0); i<mConfig.nbGenTeach; i++)
		{
			
            Eigen::MatrixXf noiseInput = Eigen::MatrixXf::Random(mConfig.genLayerNbChannels[0],mConfig.genLayerSizes[0]);
			Eigen::MatrixXf input = mGenerator->processNetwork(noiseInput);
            noiseInput = Eigen::MatrixXf::Random(mConfig.genLayerNbChannels[0],mConfig.genLayerSizes[0]);
			input = mGenerator->processNetwork(noiseInput);
			
			desiredOutput(0,0) = 1;
			mTeacher.backpropGenerator(input, desiredOutput, mConfig.step, mConfig.dx);
		}
        for(unsigned int i(0); i<mConfig.nbDisTeach; i++)
		{
            noiseInput = Eigen::MatrixXf::Random(mConfig.genLayerNbChannels[0],mConfig.genLayerSizes[0]);
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
    int i(limit);
    float errorMean{0};
    if (returnErrorRate)
    {
        for(std::vector<Sample>::iterator itr = mTestingBatchDis.begin(); itr != mTestingBatchDis.end() && i-- != 0; ++itr)
        {
            Eigen::MatrixXf output{mDiscriminator->processNetwork(itr->first)};
            errorMean += sqrt((output).squaredNorm());
        }
    }
    return errorMean/static_cast<float>(std::min((int)mTestingBatchDis.size(),limit));
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

/*    auto layersSizesDis = document["layersSizesDis"].GetArray();
    for(rapidjson::SizeType i = 0; i < layersSizesDis.Size(); i++)
        mConfig.disLayerSizes.push_back(layersSizesDis[i].GetUint());

    auto layersSizesGen = document["layersSizesGen"].GetArray();
    for(rapidjson::SizeType i = 0; i < layersSizesGen.Size(); i++)
        mConfig.genLayerSizes.push_back(layersSizesGen[i].GetUint());*/

    mConfig.databaseToUse = document["databaseToUse"].GetString();
    if(mConfig.databaseToUse == "mnist")
        mConfig.imageSizeSide = 28;
    else if(mConfig.databaseToUse == "cifar10")
        mConfig.imageSizeSide = 32;

    auto chiffresATracer = document["chiffreATracer"].GetArray();
    for(rapidjson::SizeType i(0); i < chiffresATracer.Size(); i++)
        mConfig.chiffresATracer.push_back(chiffresATracer[i].GetUint());

    auto classesCifar = document["classesCifar"].GetArray();
    for(rapidjson::SizeType i(0); i < classesCifar.Size(); i++)
        mConfig.classesCifar.push_back(classesCifar[i].GetString());

/*    auto layersTypesDis = document["layersTypesDis"].GetArray();
    for(rapidjson::SizeType i = 0; i < layersTypesDis.Size(); i++)
        mConfig.disLayerTypes.push_back(layersTypesDis[i].GetUint());

    auto layersTypesGen = document["layersTypesGen"].GetArray();
    for(rapidjson::SizeType i = 0; i < layersTypesGen.Size(); i++)
        mConfig.genLayerTypes.push_back(layersTypesGen[i].GetUint());*/

    auto chiffreATracer = document["chiffreATracer"].GetArray();
    for(rapidjson::SizeType i = 0; i < chiffreATracer.Size(); i++)
        mConfig.chiffresATracer.push_back(chiffresATracer[i].GetUint());

    auto layersDis = document["layersDis"].GetArray();
    for (int i(0); layersDis.Size()>i; i++)
    {
        mConfig.disLayerTypes.push_back(layersDis[i]["layerType"].GetUint());
        mConfig.disLayerSizes.push_back(layersDis[i]["inputSize"].GetUint());
        mConfig.disLayerNbChannels.push_back(layersDis[i]["inputChannels"].GetUint());
        mConfig.disLayerArgs.push_back(std::vector<unsigned int>());
        for (int j(0); (layersDis[i]["arguments"].GetArray()).Size()>j; j++)
        {
            mConfig.disLayerArgs[i].push_back(((layersDis[i].GetObject())["arguments"].GetArray())[j].GetUint());
        }
    }

    auto layersGen = document["layersGen"].GetArray();
    for (int i(0); layersGen.Size()>i; i++)
    {
        mConfig.genLayerTypes.push_back(layersGen[i]["layerType"].GetUint());
        mConfig.genLayerSizes.push_back(layersGen[i]["inputSize"].GetUint());
        mConfig.genLayerNbChannels.push_back(layersGen[i]["inputChannels"].GetUint());
        mConfig.genLayerArgs.push_back(std::vector<unsigned int>());
        for (int j(0); (layersGen[i]["arguments"].GetArray()).Size()>j; j++)
        {
            mConfig.genLayerArgs[i].push_back(((layersGen[i].GetObject())["arguments"].GetArray())[j].GetUint());
        }
    }

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
    mConfig.nbImgParIntervalleImg = document["nbImgParIntervalleImg"].GetUint();

	mConfig.minibatchSize = document["minibatchSize"].GetUint();
    mConfig.genFunction = document["genFunction"].GetUint();

    mConfig.descentTypeGen = document["descentTypeGen"].GetUint();
    mConfig.descentTypeDis = document["descentTypeDis"].GetUint();

    mConfig.generatorPath = document["generatorPath"].GetString();
    mConfig.discriminatorPath = document["discriminatorPath"].GetString();

    mConfig.generatorDest = document["generatorDest"].GetString();
    mConfig.discriminatorDest = document["discriminatorDest"].GetString();
	mConfig.CSVFileNameResult = document["CSVFileNameResult"].GetString();
	mConfig.CSVFileNameImage = document["CSVFileNameImage"].GetString();
	
	mConfig.typeOfExperiment = document["typeOfExperiment"].GetString();
	mConfig.useAverageForBatchlearning = document["useAverageForBatchlearning"].GetBool();
}

void Application::exportPoids()
{
    csvfile csvGen(mConfig.generatorDest);
    for(unsigned int i(0); i < mConfig.genLayerSizes.size(); i++)
       csvGen << mConfig.genLayerSizes[i];
    csvGen << endrow;
    //csvGen << *mGenerator;

    csvfile csvDis(mConfig.discriminatorDest);
    for(unsigned int i(0); i < mConfig.disLayerSizes.size(); i++)
       csvDis << mConfig.disLayerSizes[i];
    csvDis << endrow;
    //csvDis << *mDiscriminator;
    std::cout << "Export des réseaux effectués !" << std::endl;
}

NeuralNetwork* Application::importNeuralNetwork(std::string networkPath,Functions::ActivationFun activationFun)
{
    std::ifstream ifs (networkPath);
    std::string a;
    std::vector<Eigen::MatrixXf> neuralNetwork;
    std::vector<Eigen::MatrixXf> bias;
    std::vector<unsigned int> taille;
    unsigned int k = 0;
    getline(ifs, a,'\n');
    std::string b = "";
    for(unsigned int i(0); i < a.length(); i++)
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
    for(unsigned int i(0); i < taille.size()-1; i++)
    {
        neuralNetwork.push_back(Eigen::MatrixXf::Zero(taille[i],taille[i+1]));
        bias.push_back(Eigen::MatrixXf::Zero(1,taille[i+1]));
    }
    std::vector<Functions::ActivationFun> activationFunVector;
    for(unsigned int i(0); i < taille.size()-1; i++)
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
    return (nullptr);//new NeuralNetwork(taille, neuralNetwork, bias, activationFunVector));
}
