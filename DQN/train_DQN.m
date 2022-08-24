%% 
% DQN Model Definitions, 2022-07-29
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

%% Create Critic
dnn = [
    featureInputLayer(1,'Normalization','none','Name','state')
    fullyConnectedLayer(24,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(24, 'Name','CriticStateFC2')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(length(actInfo.Elements),'Name','output')];
dnn = dlnetwork(dnn);
    
criticOpts = rlOptimizerOptions('LearnRate',2e-04,'GradientThreshold',1);
critic = rlVectorQValueFunction(dnn,obsInfo,actInfo);

%% Create DQN Agent
agentOpts = rlDQNAgentOptions(...
    'SampleTime',0.15,...
    'CriticOptimizerOptions',criticOpts,...
    'ExperienceBufferLength',10000,... 
    'DiscountFactor',0.99);
agent = rlDQNAgent(critic,agentOpts);

%% Training Opts

trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 10000;
trainOpts.MaxStepsPerEpisode = 3000; %Set when stairs signal end
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 0;
trainOpts.ScoreAveragingWindowLength = 10;
trainOpts.SaveAgentCriteria = "EpisodeCount";
trainOpts.SaveAgentValue = 500;
trainOpts.SaveAgentDirectory = "DQNAgents_1";
trainOpts.Plots = "None";
trainOpts.Verbose = true;

%% Train Agent

trainingStats = train(agent,env,trainOpts);


