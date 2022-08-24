%% 
% DDPG Model Definitions, 2022-07-28
clc; clear;

%% Create Environment
rlModelName = "AC_model";
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
env = rlSimulinkEnv(rlModelName,"AC_model/RL Agent",obsInfo,actInfo);

%% Create Critic
criticNetwork = [
    featureInputLayer(1,'Normalization','none','Name','state')
    fullyConnectedLayer(32,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(1, 'Name', 'CriticFC')];
criticNetwork = dlnetwork(criticNetwork);

criticOpts = rlOptimizerOptions('LearnRate',2e-04,'GradientThreshold',1);
critic = rlValueFunction(criticNetwork,obsInfo);

%% Create Actor
actorNetwork = [
    featureInputLayer(1,'Normalization','none','Name','state')
    fullyConnectedLayer(32, 'Name','ActorStateFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(length(actInfo.Elements),'Name','ActorStateFC2')
    softmaxLayer('Name','actionProb')];
actorNetwork = dlnetwork(actorNetwork);

actorOpts = rlOptimizerOptions('LearnRate',2e-04,'GradientThreshold',1);

actor = rlDiscreteCategoricalActor(actorNetwork,obsInfo,actInfo);

%% Create AC Agent
agentOpts = rlACAgentOptions(...
    'ActorOptimizerOptions',actorOpts, ...
    'CriticOptimizerOptions',criticOpts,...
    'SampleTime',0.15, ...
    'DiscountFactor',0.99);

agent = rlACAgent(actor,critic,agentOpts);

%% Training Opts

trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 10000;
trainOpts.MaxStepsPerEpisode = 3000; %Set when stairs signal end
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 0;
trainOpts.ScoreAveragingWindowLength = 10;
trainOpts.SaveAgentCriteria = "EpisodeCount";
trainOpts.SaveAgentValue = 500;
trainOpts.SaveAgentDirectory = "ACAgents_1";
trainOpts.Plots = "None";
trainOpts.Verbose = true;

%% Train Agent

trainingStats = train(agent,env,trainOpts);


