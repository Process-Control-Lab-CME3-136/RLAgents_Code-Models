%% 
% DDPG Model Definitions, 2022-07-28
clc; clear;

%% Create Environment
rlModelName = "DDPG_model";
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
env = rlSimulinkEnv(rlModelName,"DDPG_model/RL Agent",obsInfo,actInfo);

%% Create Critic
statePath = [
    featureInputLayer(1,'Normalization','none','Name',"state")
    fullyConnectedLayer(50,'Name',"CriticStateFC1")
    reluLayer('Name', "CriticRelu1")
    fullyConnectedLayer(25,'Name',"CriticStateFC2")];
actionPath = [
    featureInputLayer(1,'Normalization','none','Name',"action")
    fullyConnectedLayer(25,'Name',"CriticActionFC1",'BiasLearnRateFactor',0)];
commonPath = [
    additionLayer(2,'Name',"add")
    reluLayer('Name',"CriticCommonRelu")
    fullyConnectedLayer(1,'Name',"CriticOutput")];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
    
criticNetwork = connectLayers(criticNetwork,"CriticStateFC2",'add/in1');
criticNetwork = connectLayers(criticNetwork,"CriticActionFC1",'add/in2');
criticNetwork = dlnetwork(criticNetwork);

criticOpts = rlOptimizerOptions('LearnRate',2e-04,'GradientThreshold',1);
critic = rlQValueFunction(criticNetwork,obsInfo,actInfo, ...
    'ObservationInputNames',"state",'ActionInputNames',"action");

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

%% Create DDPG Agent
agentOpts = rlDDPGAgentOptions(...
    'SampleTime',0.15,...
    'CriticOptimizerOptions',criticOpts,...
    'ActorOptimizerOptions',actorOpts,...
    'ExperienceBufferLength',10000,... %10000
    'DiscountFactor',0.99);

agent = rlDDPGAgent(actor,critic,agentOpts);

%% Training Opts

trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 10000;
trainOpts.MaxStepsPerEpisode = 4000; %Set when stairs signal end
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 0;
trainOpts.ScoreAveragingWindowLength = 10;
trainOpts.SaveAgentCriteria = "EpisodeCount";
trainOpts.SaveAgentValue = 500;
trainOpts.SaveAgentDirectory = "DDPGAgents_1";
trainOpts.Plots = "None";
trainOpts.Verbose = true;

%% Train Agent

trainingStats = train(agent,env,trainOpts);


