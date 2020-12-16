# Adversarial Meta Sampling for Multilingual Low-Resource Speech Recognition

*Code coming soon...*

This repository is the official implementation of the following paper:

Adversarial Meta Sampling for Multilingual Low-Resource Speech Recognition

*Yubei Xiao, Ke Gong, Pan Zhou, Guolin Zheng, Xiaodan Liang, Liang Lin. AAAI 2021.*


## Requirements
- Python 3
- Computing power (high-end GPU) and memory space (both RAM/GPU's RAM) is **extremely important** if you'd like to train your own model.
- Required packages are listed [here](requirements.txt).

To install requirements:

```setup
pip install -r requirements.txt
```


## Datasets
You can download the Common Voice corpus for free here: [Common Voice](https://commonvoice.mozilla.org/en/languages).

You can buy the IARPA BABEL dataset through [LDC](https://www.ldc.upenn.edu/)(eg. [Bengali Language Pack](https://catalog.ldc.upenn.edu/LDC2016S08)). 


## Instructions

***Before you start, make sure all the [packages required](requirements.txt) were installed correctly***

### Step 0. Preprocessing - Acoustic Feature Extraction & Text Encoding
Preprocessing scripts are placed under [`data/`](data/), you may execute them directly. The extracted data, which is ready for training, will be stored in the same place by default. For example,
```
cd data/
python3 preprocess_multilingual_corpus.py --data_path <path to Multilingual Dataset on your computer>
```

The parameters available for these scripts are as follow,

| Options        | Description                                                               |
|-------------------|---------------------------------------------------------------------------|
| data_path         | Path to the raw dataset (can be obtained by download & unzip)                 |
| feature_type      | Which type of acoustic feature to be extracted, fbank or mfcc                                                             |
| feature_dim       | Feature dimension, usually depends on the feature type (e.g. 13 for mfcc) |
| apply_delta       | Append delta of the acoustic feature to itself                            |
| apply_delta_delta | Append delta of delta                                                     |
| apply_cmvn        | Normalize acoustic feature                                                |
| output_path       | Output path for extracted data (by default, it's data/)                   |
| target            | Text encoding target, one of phoneme/char/subword/word                    |

You may check the parameter type and default value by using the option ```--help``` for each script.

### Step 1. Configuring - Model Design & Hyperparameter Setup

All the parameters related to training/decoding will be stored in a yaml file. Hyperparameter tuning and massive experiment and can be managed easily this way. See [documentation and examples](config/) for the exact format. 

### Step 2. Training - Multilingual ASR Training

Once the config file is ready, run the following command to train AMS (MML-ASR) model on source languages in the paper:

```train
python main.py --config config/meta_example.yaml \
 --MAMLSumTaskTrainer --SampleAttenController
```

To adapt the pre-trained model to target language, run this command:

```adaptation
python main.py --config config/common_voice_target_kyrgyz.yaml \
--load 0 --ckpt asr_ckpt_to_adapt
```

All settings will be parsed from the config file automatically to start training, the log file can be accessed through TensorBoard.

There are also some options,  which do not influence the performance (except `seed`), are available in this phase including the followings

| Options | Description                                                                                   |
|---------|-----------------------------------------------------------------------------------------------|
| config  | Path of config file                                                                           |
| seed    | Random seed, **note this is an option that affects the result**                                         |
| name    | Experiments for logging and saving model, by default it's <name of config file>_<random seed> |
| logdir  | Path to store training logs (log files for tensorboard), default `log/`                                                   |
| ckpdir  | Path to store results, default `result/<name>`                                                |
| njobs   | Number of workers for Pytorch DataLoader                                                      |
| cpu     | CPU-only mode, not recommended                                                                |
| no-msg  | Hide all message from stdout                                                                  |
| rnnlm   | Switch to rnnlm training mode                                                               |

### Step 3. Testing - Speech Recognition & Performance Evaluation
Once a model was trained, run the following command to test it

```
python3 main.py --config <path of config file> --test
```
Recognition result will be stored at `result/<name>/` as a txt file with auto-naming according to the decoding parameters specified in config file. The result file may be evaluated with `eval.py`. For example, test the adapted ASR on Kyrgyz of Diversity11 and check performance with:
```eval
python main.py --test --config config/common_voice_target_kyrgyz.yaml --njobs 6 --ckpt adapted_asr_ckpt --name test_kyrgyz
```

#### Check WER/CER
```
python eval.py --file result/test_kyrgyz/decode_kyrgyz_test_result.txt
```

**Notice that the meaning of options for `main.py` in this phase will change**

| Options | Description                                                                                   |
|---------|-----------------------------------------------------------------------------------------------|
| test    | *Must be enabled*|
| config  | Path of config file                                                                           |
| name    |*Must be identical to the models training phase*  |
| ckpdir  | *Must be identical to the models training phase*                                               |
| njobs   | Number of threads used for decoding, very important in terms of efficiency. Large value equals fast decoding yet RAM/GPU RAM expensive.    |
| cpu     | CPU-only mode, not recommended                                                     |
| no-msg  | Hide all message from stdout                                                              |

Rest of the options are *ineffective*  in the testing phase.

## Acknowledgements
Parts of the implementation refer to [E2E ASR](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch), a great end-to-end ASR model project by Alexander et al.