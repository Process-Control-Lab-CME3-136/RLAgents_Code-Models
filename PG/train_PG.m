%% 
% PG Model Definitions, 2022-07-28
clc; clear;

%% Create Environment
rlModelName = "PG_model";
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
env = rlSimulinkEnv(rlModelName,"PG_model/RL Agent",obsInfo,actInfo);

%% Create Actor
% input path layers
inPath = [ featureInputLayer(prod(obsInfo.Dimension), ...
              'Normalization','none','Name','netObsIn')
           fullyConnectedLayer(prod(actInfo.Dimension), ...
              'Name','infc') ];

% path layers for mean value 
% using scalingLayer to scale range from (-1,1) to (-10,10)
meanPath = [ tanhLayer('Name','tanhMean');
             fullyConnectedLayer(prod(actInfo.Dimension));
             scalingLayer('Name','scale', ...
                'Scale',actInfo.UpperLimit) ];

% path layers for standard deviations
% using softplus layer to make them non negative
sdevPath = [ tanhLayer('Name','tanhStdv');
             fullyConnectedLayer(prod(actInfo.Dimension));
             softplusLayer('Name','splus') ];

% add layers to network object
net = layerGraph(inPath);
net = addLayers(net,meanPath);
net = addLayers(net,sdevPath);

% connect layers
net = connectLayers(net,'infc','tanhMean/in');
net = connectLayers(net,'infc','tanhStdv/in');

actorNetwork = dlnetwork(net);
% Actor Options
actorOpts = rlOptimizerOptions('LearnRate',2e-04,'GradientThreshold',1);
actor = rlContinuousGaussianActor(actorNetwork,obsInfo,actInfo, ...
    ActionMeanOutputNames='scale',ActionStandardDeviationOutputNames='splus');

%% Create PG Agent
agentOpts = rlPGAgentOptions(...
    'SampleTime',0.15,...
    'ActorOptimizerOptions',actorOpts,...
    'DiscountFactor',0.99, ...
    'UseBaseline',0);
    
agent = rlPGAgent(actor,agentOpts);

%% Training Opts

trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 15000;
trainOpts.MaxStepsPerEpisode = 3000; %Set when stairs signal end
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 0;
trainOpts.ScoreAveragingWindowLength = 10;
trainOpts.SaveAgentCriteria = "EpisodeCount";
trainOpts.SaveAgentValue = 500;
trainOpts.SaveAgentDirectory = "PGAgents_1";
trainOpts.Plots = "None";
trainOpts.Verbose = true;

%% Train Agent

trainingStats = train(agent,env,trainOpts);


