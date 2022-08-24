%% Code Header
%{
Off-line agent for flow rate control, random integers training
Date Created: 2022-07-06
Author: Bowen Liu
Description: This is the agent for flow rate control scheme using
Q-learning.
%}
% Get Started with RL toolbox:
% https://www.mathworks.com/help/reinforcement-learning/getting-started-with-reinforcement-learning-toolbox.html
%%
clear;
clc;
%% Create Environment Interface
mdl = "rl_random_eu_2"; % RL Simulink model file path
open_system(mdl); % open the Simulink model

% For discrete S and A spaces, use rlFiniteSetSpec().
% For continous S and A spaces, use rlNumericSpec().

% Define S and A spaces
stateSpace = -151400:3785:151400; % Flow rate is discretized at 3.785 L/min
actionSpace = -10:1:10; % Enlarged Action Space

% Setup S space
obsInfo = rlFiniteSetSpec(stateSpace);
obsInfo.Name = 'Flow'; % .Name and .Description are set arbitrarily
numObservations = obsInfo.Dimension(1); 
% How many states are is involved?
% -In this case, only flow rate. Hence # of observations is 1.

% Setup A space
actInfo = rlFiniteSetSpec(actionSpace);
actInfo.Name = 'Motor Speed';
numActions = actInfo.Dimension(1);
% How many actions are is involved?
% -In this case, only motor speed can be changed, hence 1.

% Setup environment
env = rlSimulinkEnv(mdl,'rl_random_eu_2/RL Agent',obsInfo,actInfo);
% ResetFcn can be set

%% Create Q-learning Agent
% https://www.mathworks.com/help/reinforcement-learning/ug/q-agents.html
% 1. Create a critic using an rlQValueFunction object.
qTable = rlTable(obsInfo,actInfo);
critic = rlQValueFunction(qTable,obsInfo,actInfo);

% 2. Specify agent options using an rlQAgentOptions object.
% Learning Rate:alpha; Discount Factor:gamma
agentOpts = rlQAgentOptions;
agentOpts.SampleTime = 0.15;
agentOpts.DiscountFactor = 0.99;
agentOpts.EpsilonGreedyExploration.Epsilon = 0.02;
agentOpts.EpsilonGreedyExploration.EpsilonMin = 0.01;
agentOpts.CriticOptimizerOptions.LearnRate = 0.00025;

% 3. Create the agent using an rlQAgent object.
agent = rlQAgent(critic,agentOpts);

%% Training Options
trainOpts = rlTrainingOptions;
trainOpts.MaxEpisodes = 10000;
trainOpts.MaxStepsPerEpisode = 2000; %Set when stairs signal end
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 0;
trainOpts.ScoreAveragingWindowLength = 10;
trainOpts.SaveAgentCriteria = "EpisodeCount";
trainOpts.SaveAgentValue = 500;
trainOpts.SaveAgentDirectory = "savedQAgents_random_eu9";
trainOpts.Plots = "None";
trainOpts.Verbose = true;

%% Train or simulate the Q-learning Agent

trainingStats = train(agent,env,trainOpts);
















