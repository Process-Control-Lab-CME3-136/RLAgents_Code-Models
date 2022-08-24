%% 
% MBPO Model using SAC, 2022-08-18
clc; clear;

%% Create Environment
rlModelName = "SAC_model";
open_system(rlModelName);

% Setup States
obsInfo = rlNumericSpec([1 1],'LowerLimit',-150,'UpperLimit',150);
obsInfo.Name = "Observation";
obsInfo.Description = "Flow rate error.";
numObservations = obsInfo.Dimension(1);

actInfo = rlNumericSpec([1 1],'LowerLimit',-10,'UpperLimit',10);
actInfo.Name = "Action";
actInfo.Description = "Pump acceleration (Hz/s)";
numActions = numel(actInfo);

% Define Environment
env = rlSimulinkEnv(rlModelName,"SAC_model/RL Agent",obsInfo,actInfo);

% Create SAC Base Agent
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

%% Create SAC Agent
agentOpts = rlSACAgentOptions(...
    'ActorOptimizerOptions',actorOpts, ...
    'CriticOptimizerOptions',criticOpts,...
    'SampleTime',0.15, ...
    'DiscountFactor',0.99);

baseagent = rlSACAgent(actor,critic,agentOpts);

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
MBPOAgentOpts.IsDoneOptimizerOptions.LearnRate = 0; %% We don't need a is done net work
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
trainOpts.SaveAgentDirectory = "MBPO_SAC_Agents_1";
trainOpts.Plots = "None";
trainOpts.Verbose = true;

%% Train Agent
trainingStats = train(agent,env,trainOpts);