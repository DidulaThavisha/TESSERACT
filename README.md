# VIP CUP 2023 OLIVES Biomarker Detection


***

This work was done in the [Omni Lab for Intelligent Visual Engineering and Science (OLIVES) @ Georgia Tech](https://ghassanalregib.info/). 
This competition is based on the [OLIVES](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3be60b4a739b95a07a944a1a2c41e05e-Abstract-Datasets_and_Benchmarks.html) dataset published at NeurIPS 2022.
Feel free to check our lab's [Website](https://ghassanalregib.info/publications) 
and [GitHub](https://github.com/olivesgatech) for other interesting work!!!

The starter code provided was optimized and submitted to the competition by Team TESSERACT. 
***
Also, I'm very happy to announce that we have achieved global 6th Rank with a F1 score of 0.7921 
![WhatsApp Image 2023-09-09 at 14 08 52](https://github.com/DidulaThavisha/TESSERACT/assets/86177477/ded435f4-966a-4417-9830-35a11cd8f9be)

## Citation

Prabhushankar, M., Kokilepersaud, K., Logan, Y. Y., Trejo Corona, S., AlRegib, G., & Wykoff, C. (2022). Olives dataset: Ophthalmic labels for investigating visual eye semantics. Advances in Neural Information Processing Systems, 35, 9201-9216.

## Data

The data for this competition can be downloaded at ...

`Training_Biomarker_Data.csv`: Biomarker labels in the training set.

`Training_Unlabeled_Clinical_Data.xlsx`: Provides the clinical labels for all the data without biomarker information.

`test_set_submission_template.csv`: This provides the structure by which all submissions should be organized. 
This includes the image path and the associated 6 biomarkers.

PRIME_FULL and TREX DME are the training sets.

RECOVERY is the test set. The ground truth biomarker labels are held out, but the images and clinical data are provided.

## Submission

To submit please fill out the provided template using the model output for each image in the test set. 
There should be the file path followed by a one or zero for the presence or absence of each of 6 biomarkers for the associated image.

Submit this CSV file to the following server ...

## Starter Code Usage

`
python train.py --batch_size 64 --model 'unet' --dataset 'OLIVES' --epochs 150 --device 'cuda:0' --train_image_path '' --test_image_path '' --test_csv_path './csv_dir/test_set_submission_template.csv' --train_csv_path './csv_dir/Training_Biomarker_Data.csv'
`

Fill this out with the appropriate file path fields for the training and test data to train a model and produce a numpy 
that can act as a valid submission once the file paths are appended and saved as a CSV.

## Baseline Results

With this repository, <mark>a ResNet-50 model, and 100 epochs of training</mark>, we achieved a macro-averaged F1-Score of .6256.

## Phase 2 Evaluation 

We have included a code demo called `phase2_sample_eval.py`. It shows the process by which macro f1-scores are computed for individual 
patients in this phase of the challenge. The main difference is that instead of averaging f1-scores across all images, we now average across
each patient to get a f1-score for each patient. Teams will be assessed on how well they do on all patients, rather than performing very well
on some and poorly on others.

Please note that the submission process is the exact same as phase 1, but we want to show you how we will now process your submission files differently.

### Acknowledgements

This work was done in collaboration with the [Retina Consultants of Texas](https://www.retinaconsultantstexas.com/).
This codebase utilized was partly constructed with code from the [Supervised Contrastive Learning](https://github.com/HobbitLong/SupContrast) Github.
