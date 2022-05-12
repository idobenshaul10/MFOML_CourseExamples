# SVSL Code

This code accompanies the paper **"Nearest Class-Center Simplification through Intermediate Layer"**.
With it, the user can run training runs for all datasets in the paper, along with the option to run using the SVSL loss function. All training and intermediate-layer metrics are pushed to Weights and Biases(part of the installation). Please see requirements.txt for installation. 


# Vision
All experiments are based on the `Vision/configs`  folder. Example command:

    python Vision/neuralcollapse_dist.py 1e-5 0 True Vision/configs/STL10_Resnet50.p

where `1e-5` is the **α** parameter and `0` is the **γ** hyper parameter. The True/False controls whether to use the SVSL loss or not. There is a config file for each vision experiment show in the paper. There is 

# NLP
In order to run the NLP experiments, one has to run the: 

    python NLP/train_glue.py --task_name cola --use_consistency_loss
where: 

 - task_name  - which task to run the experiment on: 'cola', 'rte', 'mrpc', 'sst2'
 - use_consistency_loss - whether to use SVSL loss. 
 The **α** and **γ** parameters are to be changed in `NLP/train_glue.py` main function(like so): `next_parameters = {'consis_param': 1e-5, 'first_layer_used': 0}`
 
**Scripts for plotting graphs will be uploaded upon release.**
