%% 
% PPO Model Definitions, 2022-08-08
clc; clear;

%% Create Environment
rlModelName = "PPO_model";
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
env = rlSimulinkEnv(rlModelName,"PPO_model/RL Agent",obsInfo,actInfo);

%% Create Critic
criticNet = [
    featureInputLayer(prod(obsInfo.Dimension), ...
        'Normalization','none','Name','state')
    fullyConnectedLayer(10,'Name', 'fc_in')
    reluLayer('Name', 'relu')
    fullyConnectedLayer(1,'Name','out')];
critic = rlValueFunction(criticNet,obsInfo);
criticOpts = rlOptimizerOptions( ...
    'LearnRate',0.0002,'GradientThreshold',1);

%% Create Actor
% input path layer
inPath = [ 
    featureInputLayer(prod(obsInfo.Dimension), ...
        'Normalization','none','Name','state')
    fullyConnectedLayer(10,'Name', 'ip_fc')
    reluLayer('Name', 'ip_relu')
    fullyConnectedLayer(1,'Name','ip_out') ];

% path layers for mean value
meanPath = [
    fullyConnectedLayer(15,'Name', 'mp_fc1')
    reluLayer('Name', 'mp_relu')
    fullyConnectedLayer(1,'Name','mp_fc2');
    tanhLayer('Name','tanh');
    scalingLayer('Name','mp_out', ...
         'Scale',actInfo.UpperLimit) ]; % range: (-2N,2N)

% path layers for standard deviation 
sdevPath = [
    fullyConnectedLayer(15,'Name', 'vp_fc1')
    reluLayer('Name', 'vp_relu')
    fullyConnectedLayer(1,'Name','vp_fc2');
    softplusLayer('Name', 'vp_out') ]; % range: (0,+Inf)

% add layers to layerGraph network object
actorNet = layerGraph(inPath);
actorNet = addLayers(actorNet,meanPath);
actorNet = addLayers(actorNet,sdevPath);

% connect layers
actorNet = connectLayers(actorNet,'ip_out','mp_fc1/in');
actorNet = connectLayers(actorNet,'ip_out','vp_fc1/in');

actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
    'ActionMeanOutputNames','mp_out',...
    'ActionStandardDeviationOutputNames','vp_out',...
    'ObservationInputNames','state');

%% Create Agent
agentOpts = rlPPOAgentOptions(...
    'SampleTime',0.15,...
    'ExperienceHorizon',500, ...
    'DiscountFactor',0.99, ...
    'CriticOptimizerOptions',criticOpts);
agent = rlPPOAgent(actor,critic,agentOpts);

%% Training Opts

trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 10000;
trainOpts.MaxStepsPerEpisode = 3000; %Set when stairs signal end
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 0;
trainOpts.ScoreAveragingWindowLength = 10;
trainOpts.SaveAgentCriteria = "EpisodeCount";
trainOpts.SaveAgentValue = 500;
trainOpts.SaveAgentDirectory = "PPOAgents_2";
trainOpts.Plots = "None";
trainOpts.Verbose = true;

%% Train Agent

trainingStats = train(agent,env,trainOpts);


