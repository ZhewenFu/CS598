# CS 598 DLH Final Project

## Original Paper
Xueping Peng, Guodong Long, Tao Shen, Sen Wang,Jing Jiang, and Chengqi Zhang. 2020. Bitenet: Bidi-rectional temporal encoder network to predict medical outcomes.

## Link to Original Paper's Repo
[BiteNet](https://github.com/Xueping/BiteNet)

## Dependencies
Python2.7, Python3, Theano, Sklearn 

## Data Download and Preparation
1. For the BiteNet model, it uses dataset.data_prepararion.py for MIMIC III dataset. We have already the data preprocessing for this and stored the output as a json file under BiteNet/dataset/processed, so no need to run it again.

2. For the RETAIN model, it use process_mimic.py to preprocess the data. We have put every dataset we need to perform the data preprocessing (i.e. ADMISSION.csv, DIAGNOSES_ICD.csv and PATIENTS.csv).

3. ## IMPORTANT NOTE to TAs: we have deleted all the MIMIC data and processed MIMIC data from our repo, if you need to run the code, you will have to contact us to obtain the processed data.

## Data Preprocessing and Model Training Command

**STEP 1: Install Packages**  
1. The experiment requires both python2 (for RETAIN code) and python3 (for BiteNet code)

2. Install Theano and sklearn for python2. Input command
`pip2 install theano`
`pip2 install sklearn`

**STEP 2: Test BiteNet**  
1. Change the path to current folder.

2. Use BiteNet.train.BiteNet_mh_DX.py for future diagnosis prediction. Input command 
`python BiteNet_mh_DX.py --data_source mimic3  --model Bite --verbose True --task BiteNet --predict_type dx --visit_threshold 2 --max_epoch 5 --train_batch_size 32 --gpu 0 --valid_visits 10 --num_hidden_layers 1 --pos_encoding encoding --min_cut_freq 5 --embedding_size 130 --dropout 0.1 --only_dx_flag False`

3. Use BiteNet.train.BiteNet_mh_RE.py for future re-admission prediction. Input command 
`python BiteNet_mh_RE.py --data_source mimic3  --model Bite --verbose True --task BiteNet --predict_type re --visit_threshold 2 --max_epoch 5 --train_batch_size 32 --gpu 0 --valid_visits 10 --num_hidden_layers 1 --pos_encoding encoding --min_cut_freq 5 --embedding_size 130 --dropout 0.1 --only_dx_flag False`

4. The corresponding evaluation results will show in stdout.

**STEP 3: Test RETAIN**  
1. Change the path to folder Baseline/RETAIN. Input command 
`cd Baseline/RETAIN`

2. Use process_mimic.py to preprocess the data. Input command
`python2 process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv PATIENTS.csv output`

3. Use retain.py to train the model. Input command
`python2 retain.py output.3digitICD9.seqs 942 output.readmission output --simple_load --n_epochs 5 --keep_prob_context 0.8 --keep_prob_emb 0.5`

4. The corresponding evaluation results will show in stdout.

## Table of results

**BiteNet**  
1. For diagnosis prediction, we have:

| Diagnosis at k ||||||
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| k=5  | k=10 |  k=15 |  k=20 |  k=25 |  k=30 |
| 0.648 | 0.6024 | 0.6463 | 0.7076 | 0.7595 | 0.8073|

2. For readmission prediction, we have:

| accuracy | precision | sensitivity |  specificity |  f_score | pr_auc  |  roc_auc  |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |  ------------- |
| 0.7931 | 0.3929 | 0.1538 | 0.9439 | 0.2211 | 0.3295 | 0.5814 |


**RETAIN**  
1. For the RETAIN model, The best validation & test AUC:0.360153, 0.367125

## Model Parameters

Below are the parameters that the origin BiteNet paper claimed to use.

--data_source mimic3 

--model Bite 

--verbose True 

--task BiteNet 

--predict_type dx 

--visit_threshold 2  

--max_epoch 5 

--train_batch_size 32 

--gpu 2 

--valid_visits 10 

--num_hidden_layers 1 

--pos_encoding encoding 

--min_cut_freq 5 

--embedding_size 128 

--dropout 0.1 

--only_dx_flag False
