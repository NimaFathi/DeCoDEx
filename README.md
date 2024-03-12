
# DeCoDEx: Deep Counterfactual Explanation Framework

Welcome to the official repository of the DeCoDEx project. This framework is designed for generating deep counterfactual explanations using diffusion models and classifiers. Below you'll find instructions on how to create datasets, train models, and generate counterfactuals, along with information on the metrics used for evaluation.

## Table of Contents
- [Create Virtual Environment](#install-venv)
- [Create Datasets](#create-datasets)
- [Train DDPM](#train-ddpm)
- [Train Classifiers](#train-classifiers)
- [Counterfactual Generation](#counterfactual-generation)
- [Metrics](#metrics)

## Install Venv
Create a virtual Environment and install the nessecary packages from the `requirements.txt` file as shown:
```bash
pip install -r requirements.txt --no-cache
```

## Create Datasets

In this project, we utilize a specific dataset format to ensure consistency and reproducibility in our experiments. To prepare your dataset in the same manner, please follow the steps outlined below:

1. **Data Preparation**: Ensure your data is in the required format. Download cheXpert dataset from this [link](https://www.kaggle.com/datasets/willarevalo/chexpert-v10-small). usee the `train.csv` file to contrive different version of the dataset.
2. **Preprocessing**: Apply the necessary preprocessing steps. We have two different datasets:
- Dot Dataset: For Dot Dataset we only use the subjects without support devices based on the labels in the `train.csv` file. 90% of the subjects with `Pleural Effusion` are augmented with the artifact whereas only 10% of subjects with `No Finding` contain the artifact. For more information check this [Notebook](notebooks/create_dot_dataset.ipynb).
- Device Dataset: For Device Dataset we use the original images and contrive the number of samples in each subgroup. For subjects with `Pleural Effusion` we contrive the number of samples in a way that 90% of such subjects also have `Support Devices` whereas for the subjects with `No Finding` statistics are the opposite (90% of such subjects does not have `Support Devices`). You can refer to this [Notebook](notebooks/create_md_dataset.ipynb) and follow the steps.

3. **Dataset Notebook**: For detailed instructions, refer to the provided Jupyter notebooks. [Dot Dataset](notebooks/create_dot_dataset.ipynb), [Device Dataset](notebooks/create_md_dataset.ipynb)

## Train DDPM

To train the Denoising Diffusion Probabilistic Model (DDPM) used in our framework, follow the steps below. Adjust the parameters as needed for your specific use case.

```bash
python train_ddpm.py --dataset [YOUR_DATASET] --epochs 50 --learning_rate 1e-4 --other_args
```

(Replace `[YOUR_DATASET]` with the path to your dataset and adjust other arguments as necessary.)

To update the classifier training section in your `README.md` with the new parser options and a more generalized command example, you could write something like this:

## Train Classifiers

This section provides instructions on how to train classifiers or detectors for the specified dataset. The training script is highly configurable with several command-line arguments to suit your training needs.

### Usage

To train your model, use the following command structure. Replace the placeholder values with your specific configurations:

```bash
python train_classifier.py \
  --data_dir [path_to_data] \
  --model_path [path_to_save_model] \
  --epochs [num_epochs] \
  --batch_size [batch_size] \
  --image_size [image_dim] \
  --lr [learning_rate] \
  --lr_sf [lr_scheduler_factor] \
  --lr_patience [lr_scheduler_patience] \
  --task [classification/detection] \
  [additional_arguments]
```

### Arguments Description
`erm.py` for ERM and `groupdro.py` for Group-DRO methods

- `--data_dir`: The directory where your dataset is located.
- `--model_path`: Path where the trained model will be saved.
- `--epochs`: Number of training epochs. Default is 40.
- `--batch_size`: Number of samples per training batch. Default is 32.
- `--image_size`: The height and width of the images in pixels. Default is 256.
- `--lr`: The initial learning rate. Default is 0.0002.
- `--lr_sf`: Factor by which the learning rate is reduced. Default is 0.1.
- `--lr_patience`: Number of epochs with no improvement after which learning rate will be reduced. Default is 5.
- `--task`: Specifies the task for the model, either 'classification' or 'detection'.
- `--random_crop`: Enable random cropping as a data augmentation method (optional).
- `--random_flip`: Enable random horizontal flipping as a data augmentation method (optional).
- `--gpu_id`: The ID of the GPU to use. Default is 0.
- `--biased`: Use this flag if you are working with a biased dataset (optional).
- `--balanced`: Use this flag if you want to use a balanced dataset (default is balanced).
- `--balance_ratio`: The ratio of positive to negative samples in the balanced dataset. Default is 0.1.
- `--augment`: Augment dataset with counterfactually generated samples.
- `--augmented_data_dir`: Path to the directory that contains CF images for augmentation.
- `--dataset`: Choose between 'PE90DotNoSupport' and 'MedicalDevicePEDataset' for the dataset.

Adjust these parameters according to your dataset and training preferences. For further customization, you can add additional arguments as needed.

## Counterfactual Generation

After training the DDPM and classifiers, you can generate counterfactuals by running:

```bash
python generate_counterfactuals.py --model_path [DDPM_MODEL_PATH] --detector_path [DETECTOR_MODEL_PATH] --classifier_path [CLASSIFIER_MODEL_PATH] --input [INPUT_DATA] --output [OUTPUT_PATH]
```

(Replace placeholders with the appropriate file paths and data.)

## Results
The directory structure for the results of the experiments is organized as follows:
```
output_path/
└── Results/
    └── exp_name/
        ├── CC/
        │   ├── CCF/
        │   │   ├── CD/
        │   │   │   ├── CF/
        │   │   │   ├── Noise/
        │   │   │   ├── Info/
        │   │   │   └── SM/
        │   │   └── ID/
        │   │       ├── CF/
        │   │       ├── Noise/
        │   │       ├── Info/
        │   │       └── SM/
        │   └── ICF/
        │       ├── CD/
        │       │   ├── CF/
        │       │   ├── Noise/
        │       │   ├── Info/
        │       │   └── SM/
        │       └── ID/
        │           ├── CF/
        │           ├── Noise/
        │           ├── Info/
        │           └── SM/
        └── IC/
            ├── CCF/
            │   ├── CD/
            │   │   ├── CF/
            │   │   ├── Noise/
            │   │   ├── Info/
            │   │   └── SM/
            │   └── ID/
            │       ├── CF/
            │       ├── Noise/
            │       ├── Info/
            │       └── SM/
            └── ICF/
                ├── CD/
                │   ├── CF/
                │   ├── Noise/
                │   ├── Info/
                │   └── SM/
                └── ID/
                    ├── CF/
                    ├── Noise/
                    ├── Info/
                    └── SM/
```

### Terminology and Structure Details

- **CC**: Correct Classifier
- **IC**: Incorrect Classifier
- **CCF**: Correct Counterfactual
- **ICF**: Incorrect Counterfactual
- **CF**: Contains the final generated counterfactual image (X), starting from `0000000.jpg` onward.
- **Noise**: Contains the noisy image (Z), starting from `0000000.jpg` onward.
- **SM**: Contains the Saliency Map showing the difference from CF image and original image, starting from `0000000.jpg` onwards.

### Info Directory Structure

The `Info` directory contains essential information needed for running the metrics, structured as follows:

- **det label**: Ground truth label for artifact.
- **det pred (org img)**: Predicted value of the detector on the original image.
- **det target**: Target value for prediction during inference; for detection, it should be the same as `det pred (org img)`.
- **cf det pred**: Predicted artifact value for the counterfactual image.
- **class label**: Ground truth label for disease.
- **class pred (org img)**: Predicted value of the classifier on the original image.
- **class target**: Target value for prediction during inference; for classification, it is always the opposite of `class pred (org img)`.
- **cf pred**: Predicted disease value for the counterfactual image.

## Metrics

To evaluate the generated counterfactuals, we employ several metrics. Detailed instructions and a sample notebook will be provided for each metric. Ensure to use the corresponding evaluation script for your analyses.

- **Metric 1**: Description and usage.
- **Metric 2**: Description and usage.
- (Add more metrics as necessary.)

For a practical demonstration on how to use these metrics, refer to the included Jupyter notebook. (Link to the notebook).
```
