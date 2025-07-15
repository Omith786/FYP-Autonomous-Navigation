**Folder Structure:**



* FYP\_Simulation\_UNITY/ – Here we have the simulation application to run on the Raspberry Pi (must have Ubuntu)
* models/ – all the trained models. The final models are inside here as well
* logs/ – TensorBoard and training logs.
* eval\_runs/ – here it has the images of every run done by the model that we are testing and its stats.
* checkpoints of trained models/ – Intermediate checkpoints for every step. Needed this since my pc being so slow sometimes just crashed and doing this I wouldn't lose everything in the hours taken.
* Optuna/ – Results from hyperparameter tuning.
* Physical Robot/ – Scripts for Raspberry Pi integration.



**MAIN FILES**



**Environment \& Training:**



R1\_Training\_Environment.py – This is the training environment for the models. Runs in Gymnasium.



R1\_Curriculum\_Trainer\_\[PPO/DQN/A2C].py – Curriculum-based trainers. These ones work with the config files where I use those to edit the parameters



R1\_\[PPO/DQN/A2C]\_train\_with\_best\_config.py – These ones also do curriculum training in stages but use the parameters from optuna.



Config \& Tuning:



R1\_\[PPO/DQN/A2C]\_Config.py – RL model hyperparameters. I used these initially , by changing up the paramters manually to see the changes and if any improvements.



R1\_\[PPO/DQN/A2C]\_Optuna\_Tuner.py – Optuna tuning scripts. Goes through 20.



Evaluation \& Optimisation:



R1\_Evaluate\_Model.py – Run trained models and collect results and sends them in eval runs.



R1\_Pruning\_PPO.py – Pruning for Pi deployment. I did it with PPO since its the best model.



**Classical Algorithms:**



R1\_Bug0\_Classical.py



R1\_Bug2\_Classical.py



R1\_FollowGap\_Classical.py



These are used for comparison with RL methods.

