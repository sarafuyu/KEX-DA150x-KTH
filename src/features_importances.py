#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handling and merging feature importances

This script is run separately from the main script.

:Date: 2024-06-10
:Authors: Sara Rydell, Noah Hopkins

Co-authored-by: Sara Rydell <sara.hanfuyu@gmail.com>
Co-authored-by: Noah Hopkins <nhopkins@kth.se>
"""
import pandas as pd
import numpy as np

# %% Data Loading

# Load the CSV file with the correct delimiter
feature_ranking_info = pd.read_csv('feature_ranking_info.csv', sep=',')

# Load the Excel file
targets_antibodies = pd.read_excel('202300619_TargetsAntibodies_for_python.xlsx', sheet_name='Targets and Antibodies')

# Ensure the 'name' column is treated as string
feature_ranking_info['name'] = feature_ranking_info['name'].astype(str)

# %% Data migration

genreate_data = True

if genreate_data:
    
    for i, row in feature_ranking_info.iterrows():
        name = row['name']
        hpa_id = name.split('_')[0]
        rank = row['rank']
        selected = row['selected']
        importance = row['importance']

        # Change name to HPA ID
        feature_ranking_info.at[i, 'name'] = hpa_id
        
        # Find the index of the row in targets_antibodies where Antibody ID HPA matches the HPA ID
        matching_rows = targets_antibodies[targets_antibodies['Antibody ID HPA'] == hpa_id]
        
        # If there isn't a match add false in indicator col
        if matching_rows.empty:
            feature_ranking_info.at[i, 'InTargetsAntibodies'] = False
        
        if not matching_rows.empty:
            # Add true in indicator col
            feature_ranking_info.at[i, 'InTargetsAntibodies'] = True
            
            # Get the index of the first matching row
            index_in_targets = matching_rows.index[0]
            feature_ranking_info.at[i, 'IndexInTargetsAntibodies'] = index_in_targets

            # Extract UnitProt No
            uniprot_no = targets_antibodies.at[index_in_targets, 'UniProt No']
            if isinstance(uniprot_no, str):
                feature_ranking_info.at[i, 'UniProtNo'] = uniprot_no
            else:
                feature_ranking_info.at[i, 'UniProtNo'] = 'No Data'
                
            # Extract ENSEMBL gene ID
            ensembl_gene_id = targets_antibodies.at[index_in_targets, 'ENSEMBL gene ID']
            if isinstance(ensembl_gene_id, str):
                feature_ranking_info.at[i, 'ENSEMBLGeneID'] = ensembl_gene_id
            else:
                feature_ranking_info.at[i, 'ENSEMBLGeneID'] = 'No Data'
                
            # Extract Gene description
            gene_description = targets_antibodies.at[index_in_targets, 'Gene description']
            if isinstance(gene_description, str):
                feature_ranking_info.at[i, 'GeneDescription'] = gene_description
            else:
                feature_ranking_info.at[i, 'GeneDescription'] = 'No Data'
            
            # Extract Gene name
            gene_name = targets_antibodies.at[index_in_targets, 'Gene name']
            if isinstance(gene_name, str):
                feature_ranking_info.at[i, 'GeneName'] = gene_name
            else:
                feature_ranking_info.at[i, 'GeneName'] = 'No Data'
            
            # Extract Antibody ID HPA
            antibody_id_hpa = targets_antibodies.at[index_in_targets, 'Antibody ID HPA']
            if isinstance(antibody_id_hpa, str):
                feature_ranking_info.at[i, 'AntibodyIDHPA'] = antibody_id_hpa
            else:
                feature_ranking_info.at[i, 'AntibodyIDHPA'] = 'No Data'

            # Extract the IHC validation score
            #ihc_validation = targets_antibodies.at[index_in_targets, 'IHC validation score']
            #if isinstance(ihc_validation, str):
            #    ihc_validation_score = int(ihc_validation.split(':')[0])
            #else:
            #    ihc_validation_score = ihc_validation
            #feature_ranking_info.at[i, 'IHCValidationScore'] = ihc_validation_score

            # Extract the Target selection criteria
            target_selection_criteria = targets_antibodies.at[index_in_targets, 'Target selection criteria']
            if isinstance(target_selection_criteria, str):
                feature_ranking_info.at[i, 'TargetSelectionCriteria'] = target_selection_criteria
            else:
                feature_ranking_info.at[i, 'TargetSelectionCriteria'] = 'No Data'

    # Save the updated DataFrame to a new CSV file
    feature_ranking_info.to_csv('updated_feature_ranking_info.csv', index=False)

# %% Statistik

show_stats = False

if show_stats:
        
    feature_ranking_info = pd.read_csv('updated_feature_ranking_info.csv', sep=',')
    #print(feature_ranking_info.shape) # (999, 8)

    # Number of biomarkers from all features available
    bio_all = feature_ranking_info.iloc[feature_ranking_info[feature_ranking_info['TargetSelectionCriteria'] == 'Compiled published biomarkers'].index]
    #print(bio_all.shape) # (709, 8)

    # Number of selected features
    selected_features = feature_ranking_info.iloc[feature_ranking_info[feature_ranking_info['selected'] == True].index]
    #print(selected_features.shape) # (89, 8)

    # Sum of importance values of all feature
    all_importances = selected_features['importance'].tolist()
    importance_sum = sum(map(float, all_importances))
    print(importance_sum)

    # Number of selected features that are biomarkers
    selected_features_bio = selected_features.iloc[selected_features[selected_features['TargetSelectionCriteria'] == 'Compiled published biomarkers'].index]
    #print(selected_features_bio.shape) # (66, 8)

    # Get all unique values from the 'TargetSelectionCriteria' column
    unique_target_selection_criteria = selected_features['TargetSelectionCriteria'].unique()
    # Convert to a list (optional, since unique() already returns a numpy array)
    unique_target_selection_criteria_list = list(unique_target_selection_criteria)
    for t in unique_target_selection_criteria_list:
        print(t)

    # Number of selected biomarkers that are supported 1-4
    #selected_features_bio_supported = selected_features_bio[selected_features_bio['IHCValidationScore'] <= 4.0] # selected_features_bio.iloc[selected_features_bio[selected_features_bio['IHCValidationScore'] <= 4.0].index]
    #print(selected_features_bio_supported) # (51, 8)
    
    # Number of selected features that are biomarkers
    bio_selected = len(feature_ranking_info[(feature_ranking_info['selected'] == True) & (feature_ranking_info['TargetSelectionCriteria'] == 'Compiled published biomarkers')])
    print(f"Number 'Compiled published biomarkers' among selected features: {bio_selected}")
    print(f"Share of 'Compiled published biomarkers' among selected features: {bio_selected / 89}")

    # Number of all features
    bio_all = len(feature_ranking_info[feature_ranking_info['TargetSelectionCriteria'] == 'Compiled published biomarkers'])
    print(f"Number of 'Compiled published biomarkers' among all features: {bio_all}")
    print(f"Share of 'Compiled published biomarkers' among all features: {bio_all / 89}")

    # Number of supported biomarkers
    supported_all = len(feature_ranking_info[(feature_ranking_info['TargetSelectionCriteria'] == 'Compiled published biomarkers') & (feature_ranking_info['IHCValidationScore'] <= 4)])
    print(f"Number of supported 'Compiled published biomarkers' among all features: {supported_all}")
    print(f"Share of supported 'Compiled published biomarkers' among all biomarkers: {supported_all / bio_all}")

    # Number of supported biomarkers among selected features
    supported_selected = len(feature_ranking_info[(feature_ranking_info['selected'] == True) & (feature_ranking_info['TargetSelectionCriteria'] == 'Compiled published biomarkers') & (feature_ranking_info['IHCValidationScore'] <= 4)])
    print(f"Number of supported 'Compiled published biomarkers' among selected features: {supported_selected}")
    print(f"Share of supported 'Compiled published biomarkers' among selected features: {supported_selected / bio_selected}")