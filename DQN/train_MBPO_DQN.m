%% 
% MBPO Model using DQN, 2022-08-12
clc; clear;

%% Create Environment
rlModelName = "DQN_model";
open_system(rlModelName);

actionSpace = -10:1:10;

% Setup States
obsInfo = rlNumericSpec([1 1],'LowerLimit',-150,'UpperLimit',150);
obsInfo.Name = "Observation";
obsInfo.Description = "Flow rate error.";
numObs = obsInfo.Dimension(1);

actInfo = rlFiniteSetSpec(actionSpace);
actInfo.Name = "Action";
actInfo.Description = "Pump acceleration (Hz/s)";
numActs = numel(actInfo);

% Define Environment
env = rlSimulinkEnv(rlModelName,"DQN_model/RL Agent",obsInfo,actInfo);
numObservations = 1;
numActions = 1;


%% Create DQN Base Agent
qNetwork = [
    featureInputLayer(obsInfo.Dimension(1),'Normalization','none','Name','state')
    fullyConnectedLayer(24,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(24, 'Name','CriticStateFC2')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(length(actInfo.Elements),'Name','output')];
qNetwork = dlnetwork(qNetwork);

criticOpts = rlOptimizerOptions('LearnRate',2e-04,'GradientThreshold',1);
critic = rlVectorQValueFunction(qNetwork,obsInfo,actInfo);

agentOpts = rlDQNAgentOptions(...
    'SampleTime',0.15,...
    'CriticOptimizerOptions',criticOpts,...
    'ExperienceBufferLength',10000,... 
    'DiscountFactor',0.99);
baseagent = rlDQNAgent(critic,agentOpts);

%% Create Transition Model
statePath = featureInputLayer(numObservations, ...
    Normalization="none",Name="state");
actionPath = featureInputLayer(numActions, ...
    Normalization="none",Name="action");
commonPath = [concatenationLayer(1,2,Name="concat")
    fullyConnectedLayer(64,Name="FC1")
    reluLayer(Name="CriticRelu1")
    fullyConnectedLayer(64, "Name","FC3")
    reluLayer(Name="CriticCommonRelu2")
    fullyConnectedLayer(numObservations,Name="nextObservation")];

transitionNetwork = layerGraph(statePath);
transitionNetwork = addLayers(transitionNetwork,actionPath);
transitionNetwork = addLayers(transitionNetwork,commonPath);

transitionNetwork = connectLayers(transitionNetwork,"state","concat/in1");
transitionNetwork = connectLayers(transitionNetwork,"action","concat/in2");

transitionNetwork = dlnetwork(transitionNetwork);

transitionFcn = rlContinuousDeterministicTransitionFunction( ...
    transitionNetwork,obsInfo,actInfo,...
    ObservationInputNames="state",...
    ActionInputNames="action",...
    NextObservationOutputNames="nextObservation");

%% Create Reward Fcn
actionPath = featureInputLayer(numActions,...
    Normalization="none",Name="action");
nextStatePath = featureInputLayer(numObservations,...
    Normalization="none",Name="nextState");
commonPath = [concatenationLayer(1,2,Name="concat")
    fullyConnectedLayer(64,Name="FC1")
    reluLayer(Name="CriticRelu1")
    fullyConnectedLayer(64, "Name","FC2")
    reluLayer(Name="CriticCommonRelu2")
    fullyConnectedLayer(64, "Name","FC3")
    reluLayer(Name="CriticCommonRelu3")
    fullyConnectedLayer(1,Name="reward")];

rewardNetwork = layerGraph(nextStatePath);
rewardNetwork = addLayers(rewardNetwork,actionPath);
rewardNetwork = addLayers(rewardNetwork,commonPath);

rewardNetwork = connectLayers(rewardNetwork,"nextState","concat/in1");
rewardNetwork = connectLayers(rewardNetwork,"action","concat/in2");

rewardNetwork = dlnetwork(rewardNetwork);

rewardFcn = rlContinuousDeterministicRewardFunction( ...
    rewardNetwork,obsInfo,actInfo, ...
    ActionInputNames="action",...
    NextObservationInputNames="nextState");

%% Create Is-done Fcn
commonPath = [featureInputLayer(numObservations, ...
    Normalization="none",Name="nextState");
fullyConnectedLayer(64,Name="FC1")
reluLayer(Name="CriticRelu1")
fullyConnectedLayer(64,'Name',"FC3")
reluLayer(Name="CriticCommonRelu2")
fullyConnectedLayer(2,Name="isdone0")
softmaxLayer(Name="isdone")];

isDoneNetwork = layerGraph(commonPath);
isDoneNetwork = dlnetwork(isDoneNetwork);

isdoneFcn = rlIsDoneFunction(isDoneNetwork,obsInfo,actInfo, ...
    NextObservationInputNames="nextState");

%% Create Neural Network Env
generativeEnv = rlNeuralNetworkEnvironment(obsInfo,actInfo,...
    transitionFcn,rewardFcn,isdoneFcn);

%% Create MBPO Agent
MBPOAgentOpts = rlMBPOAgentOptions; % Doesn't require sample time
MBPOAgentOpts.TransitionOptimizerOptions = rlOptimizerOptions(...
    LearnRate=1e-4,...
    GradientThreshold=1.0);
agent = rlMBPOAgent(baseagent,generativeEnv,MBPOAgentOpts);

%% Training Opts
trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 10000;
trainOpts.MaxStepsPerEpisode = 3000; %Set when stairs signal end
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 0;
trainOpts.ScoreAveragingWindowLength = 10;
trainOpts.SaveAgentCriteria = "EpisodeCount";
trainOpts.SaveAgentValue = 500;
trainOpts.SaveAgentDirectory = "MBPOAgents_1";
trainOpts.Plots = "None";
trainOpts.Verbose = true;

%% Train Agent
trainingStats = train(agent,env,trainOpts);