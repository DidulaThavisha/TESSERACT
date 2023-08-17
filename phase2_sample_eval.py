import pandas as pd
from sklearn.metrics import f1_score

# Load the CSV files
#############################################################################################################
# This script provides a sample of how the evaluation for phase 2 is different than that of phase 1.
# F1-scores are computed per patient and then averaged as a whole.
# This is assessming performance at an individual patient level, rather than a dataset as a whole.

# test_set_labels.csv refers to the labels we have access to in our interla evaluation platform
# baseline_result.csv refers to the csv file that you have submitted and is in the same format as the first phase.
#############################################################################################################
# Load the CSV files
ground_truth_df = pd.read_csv('test_set_labels.csv')
test_predictions_df = pd.read_csv('baseline_result.csv')

# Extract disease columns
disease_columns = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']

ground_truth = ground_truth_df[disease_columns].values
test_predictions = test_predictions_df[disease_columns].values

# Calculate the F1 macro scores for each image
f1_scores = []
for i in range(len(ground_truth)):
    true_labels = ground_truth[i]
    predicted_labels = test_predictions[i]
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    f1_scores.append(f1)

# Extract patient names from image names (assuming 0x-0xx format)
patient_names = [image_name.split('/')[2] for image_name in ground_truth_df['Path (Trial/Image Type/Subject/Visit/Eye/Image Name)']]

# Create a DataFrame to store patient names and their mean F1 macro scores
result_data = {'Patient_Name': patient_names, 'Mean_F1_Macro_Score': f1_scores}
result_df = pd.DataFrame(result_data)

# Group by patient name and calculate the mean F1 macro score
mean_f1_scores_by_patient = result_df.groupby('Patient_Name')['Mean_F1_Macro_Score'].mean()

# Create a DataFrame to store the aggregated F1 macro scores by patient
aggregated_f1_scores_df = pd.DataFrame({'Patient_Name': mean_f1_scores_by_patient.index, 'Mean_F1_Macro_Score': mean_f1_scores_by_patient.values})

# Save the aggregated F1 macro scores to a CSV file
aggregated_f1_scores_df.to_csv('aggregated_f1_scores_by_patient.csv', index=False)

# Calculate and display the average F1 macro score across all patients
average_f1_macro_score = mean_f1_scores_by_patient.mean()
print(f"\nAverage F1 Macro Score Across All Patients: {average_f1_macro_score:.4f}")