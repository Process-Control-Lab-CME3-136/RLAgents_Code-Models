%% 
% TD3 Model Definitions, 2022-08-08
clc; clear;

%% Create Environment
rlModelName = "TD3_model";
open_system(rlModelName);

% Setup States
obsInfo = rlNumericSpec([1 1],'LowerLimit',-150,'UpperLimit',150);
obsInfo.Name = "Observation";
obsInfo.Description = "Flow rate error.";
numObs = obsInfo.Dimension(1);

actInfo = rlNumericSpec([1 1],'LowerLimit',-10,'UpperLimit',10);
actInfo.Name = "Action";
actInfo.Description = "Pump acceleration (Hz/s)";
numActs = numel(actInfo);

% Define Environment
env = rlSimulinkEnv(rlModelName,"TD3_model/RL Agent",obsInfo,actInfo);

%% Create Critic1
statePath1 = [
    featureInputLayer(1,'Normalization','none','Name',"state")
    fullyConnectedLayer(50,'Name',"CriticStateFC1")
    reluLayer('Name', "CriticRelu1")
    fullyConnectedLayer(25,'Name',"CriticStateFC2")];
actionPath1 = [
    featureInputLayer(1,'Normalization','none','Name',"action")
    fullyConnectedLayer(25,'Name',"CriticActionFC1",'BiasLearnRateFactor',0)];
commonPath1 = [
    additionLayer(2,'Name',"add")
    reluLayer('Name',"CriticCommonRelu")
    fullyConnectedLayer(1,'Name',"CriticOutput")];

criticNetwork1 = layerGraph();
criticNetwork1 = addLayers(criticNetwork1,statePath1);
criticNetwork1 = addLayers(criticNetwork1,actionPath1);
criticNetwork1 = addLayers(criticNetwork1,commonPath1);
    
criticNetwork1 = connectLayers(criticNetwork1,"CriticStateFC2",'add/in1');
criticNetwork1 = connectLayers(criticNetwork1,"CriticActionFC1",'add/in2');
criticNetwork1 = dlnetwork(criticNetwork1);

critic1 = rlQValueFunction(criticNetwork1,obsInfo,actInfo, ...
    'ObservationInputNames',"state",'ActionInputNames',"action");
%% Critic2 and Options
statePath2 = [
    featureInputLayer(1,'Normalization','none','Name',"state")
    fullyConnectedLayer(40,'Name',"CriticStateFC1")
    reluLayer('Name', "CriticRelu1")
    fullyConnectedLayer(20,'Name',"CriticStateFC2")];
actionPath2 = [
    featureInputLayer(1,'Normalization','none','Name',"action")
    fullyConnectedLayer(20,'Name',"CriticActionFC1",'BiasLearnRateFactor',0)];
commonPath2 = [
    additionLayer(2,'Name',"add")
    reluLayer('Name',"CriticCommonRelu")
    fullyConnectedLayer(1,'Name',"CriticOutput")];

criticNetwork2 = layerGraph();
criticNetwork2 = addLayers(criticNetwork2,statePath2);
criticNetwork2 = addLayers(criticNetwork2,actionPath2);
criticNetwork2 = addLayers(criticNetwork2,commonPath2);
    
criticNetwork2 = connectLayers(criticNetwork2,"CriticStateFC2",'add/in1');
criticNetwork2 = connectLayers(criticNetwork2,"CriticActionFC1",'add/in2');
criticNetwork2 = dlnetwork(criticNetwork2);

critic2 = rlQValueFunction(criticNetwork2,obsInfo,actInfo, ...
    'ObservationInputNames',"state",'ActionInputNames',"action");

criticOpts = rlOptimizerOptions( ...
    'Optimizer','adam','LearnRate',0.0002,... 
    'GradientThreshold',1,'L2RegularizationFactor',2e-4);
%% Create Actor
actorNetwork = [
    featureInputLayer(numObs,'Normalization','none','Name',"state")
    fullyConnectedLayer(50,'Name','ActorFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(25,'Name','ActorFC2')
    reluLayer('Name','ActorRelu2')
    fullyConnectedLayer(1,'Name','ActorFC3')
    tanhLayer('Name','ActorTanh')
    scalingLayer('Name','ActorScaling','Scale',max(actInfo.UpperLimit))];
actorNetwork = dlnetwork(actorNetwork);
% Actor Options
actorOpts = rlOptimizerOptions('LearnRate',2e-04,'GradientThreshold',1);
actor = rlContinuousDeterministicActor(actorNetwork,obsInfo,actInfo);


%% Create Agent
agentOpts = rlTD3AgentOptions;
agentOpts.DiscountFactor = 0.99;
agentOpts.TargetSmoothFactor = 5e-3;
agentOpts.TargetPolicySmoothModel.Variance = 0.2;
agentOpts.TargetPolicySmoothModel.LowerLimit = -0.5;
agentOpts.TargetPolicySmoothModel.UpperLimit = 0.5;
agentOpts.CriticOptimizerOptions = criticOpts;
agentOpts.ActorOptimizerOptions = actorOpts;

agent = rlTD3Agent(actor,[critic1 critic2],agentOpts);

%% Training Opts

trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 10000;
trainOpts.MaxStepsPerEpisode = 3000; %Set when stairs signal end
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 0;
trainOpts.ScoreAveragingWindowLength = 10;
trainOpts.SaveAgentCriteria = "EpisodeCount";
trainOpts.SaveAgentValue = 500;
trainOpts.SaveAgentDirectory = "TRPOAgents_1";
trainOpts.Plots = "None";
trainOpts.Verbose = true;

%% Train Agent

trainingStats = train(agent,env,trainOpts);


