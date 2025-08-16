

# Deep Learning for Pancreatic Cancer Segmentation & Classification 

A multi-task deep learning pipeline for pancreatic cancer detection using nnUNetv2 and 300+ annotated 3D CT scans. Utilized a implement a dual-head network for image segmentation (background, pancreas, lesion) and lesion classification across 3 pancreas lesion subtypes.
##
The public sentiment analysis is done by using the VADER NLP to parse through Reddit posts relevant to the chosen stock. The dashboard will return 

- Sentiment analysis results
- Stock metrics
- Stock price plot
- Sentiment frequency plot
- Top Reddit posts sorted by sentiment scores
##
# Running the Project 
This project trains nnU-Net v2 to segment pancreas and lesions on CT, then predicts a three-class subtype from simple features measured on the predicted masks. Final output is a CSV named subtype_results.csv with columns Names and Subtype. The instructions need to be done in the IDE terminal and can be directly copied. 
## 
1. Create environment
conda create -n nnunet2 python=3.11 -y
conda activate nnunet2
pip install nnunetv2 nibabel simpleitk scikit-image pandas scikit-learn joblib torch torchvision torchaudio

2. Set nnU-Net paths for this shell
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"

3. Place data as described in the next section, then run planning and preprocessing
nnUNetv2_plan_and_preprocess -d 801 --verify_dataset_integrity -np 1

4. Train a model
nnUNetv2_train 801 2d 0 -device mps

5. Predict segmentations for test
nnUNetv2_predict -i "$nnUNet_raw/Dataset801_PancreasROI/imagesTs" \
-o "$nnUNet_results/Dataset801_PancreasROI/test_pred_2d_best" \
-d 801 -c 2d -f 0 -device mps -chk checkpoint_best.pth

6. Train the subtype classifier and write the CSV
python train_subtype_classifier.py
python predict_test_subtypes.py

7. The CSV will be written to
$nnUNet_results/Dataset801_PancreasROI/subtype_results.csv

