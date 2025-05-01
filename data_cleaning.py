import pandas as pd
import numpy as np
from unidecode import unidecode

# Replace with the path to your Excel file
file_path = "küm-16-04-2025.xlsx"

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)


# Function to clean and normalize all text values in the dataframe
def clean_and_normalize_all(df):
    # Apply the transformation to every cell in the dataframe
    for column in df.columns:
        df[column] = df[column].apply(lambda value: unidecode(str(value).strip().upper()) if isinstance(value, str) else value)
    return df

# Apply the function to normalize all text values
df = clean_and_normalize_all(df)


"""# Print unique values for each column
for column in df.columns:
    unique_vals = df[column].unique()
    print(f"{column} ({len(unique_vals)} unique):")
    print(unique_vals)
    print("-" * 50)

"""
# Function to clean the gender column
def clean_gender(gender):
    # Replace variations with 1 for 'KIZ' and 0 for 'ERKEK'
    if gender in ['KIZ', 'K']:
        return 1
    elif gender in ['ERKEK', 'E']:
        return 0
    elif gender == 0:
        return 0
    elif gender == 1:
        return 1 
    else:
        return gender 

# Apply the cleaning function to the 'gender' column
df['gender'] = df['gender'].apply(clean_gender)

# Print the cleaned dataframe
#print(df['gender'].unique())

df['underlying_risk_factor'] = df['underlying_risk_factor'].astype(str).str.upper().str.strip()

df['underlying_risk_factor'] = df['underlying_risk_factor'].replace({
    'EVET': 'VAR',
    '1': 'VAR',
    '0': 'YOK'
})

risk_dummies = pd.get_dummies(df['underlying_risk_factor'], prefix='underlying_risk')
df = pd.concat([df, risk_dummies], axis=1)
df[risk_dummies.columns] = df[risk_dummies.columns].astype(int)

def clean_fever(value):
    if value == 'VAR':
        return 1
    elif value == 'YOK':
        return 0
    elif value == 0:
        return 0
    elif value == 1:
        return 1
    else:
        return value 

# Apply it to the 'fever' column
df['fever'] = df['fever'].apply(clean_fever)

# Optional: Check result
#print(df['fever'].unique())

def clean_vomiting(value):
    if value == 'VAR':
        return 1
    elif value == 'YOK':
        return 0
    elif value == 0:
        return 0
    elif value == 1:
        return 1
    else:
        return value 

# Apply it to the 'vomiting' column
df['vomiting'] = df['vomiting'].apply(clean_vomiting)

# Optional: Check result
#print(df['vomiting'].unique())

def clean_burning_during_urination(value):
    if value == 'VAR':
        return 1
    elif value in ['YOK', 9]:
        return 0
    elif value == 0:
        return 0
    elif value == 1:
        return 1
    else:
        return value 

# Apply it to the 'burning_during_urination' column
df['burning_during_urination'] = df['burning_during_urination'].apply(clean_burning_during_urination)

# Optional: Check result
#print(df['burning_during_urination'].unique())

def clean_diarrhea(value):
    if value == 'VAR':
        return 1
    elif value == 'YOK':
        return 0
    elif value == 0:
        return 0
    elif value == 1:
        return 1
    else:
        return value 

# Apply it to the 'diarrhea' column
df['diarrhea'] = df['diarrhea'].apply(clean_diarrhea)

# Optional: Check result
#print(df['diarrhea'].unique())

def clean_blood_in_urine(value):
    if value == 'VAR':
        return 1
    elif value == 'YOK':
        return 0
    elif value == 0:
        return 0
    elif value == 1:
        return 1
    else:
        return value 

# Apply it to the 'blood_in_urine' column
df['blood_in_urine'] = df['blood_in_urine'].apply(clean_blood_in_urine)

# Optional: Check result
#print(df['blood_in_urine'].unique())

def clean_abdominal_pain(value):
    if value == 'VAR':
        return 1
    elif value == 'YOK':
        return 0
    elif value == 0:
        return 0
    elif value == 1:
        return 1
    else:
        return value 

# Apply it to the 'abdominal_pain' column
df['abdominal_pain'] = df['abdominal_pain'].apply(clean_abdominal_pain)

# Optional: Check result
#print(df['abdominal_pain'].unique())

def clean_received_antibiotics(value):
    if value == 'EVET':
        return 1
    elif value == 'HAYIR':
        return 0
    elif value == 0:
        return 0
    elif value == 1:
        return 1
    else:
        return value 

# Apply it to the 'received_antibiotics_before_arrival' column
df['received_antibiotics_before_arrival'] = df['received_antibiotics_before_arrival'].apply(clean_received_antibiotics)

# Optional: Check result
#print(df['received_antibiotics_before_arrival'].unique())

def clean_recurring_admissions(value):
    if value == 'EVET':
        return 1
    elif value in ['HAYIR', 'YOK']:
        return 0
    elif value == 0:
        return 0
    elif value == 1:
        return 1
    else:
        return value 
# Apply it to the 'does_patient_have_recurring_admissions' column
df['does_patient_have_recurring_admissions'] = df['does_patient_have_recurring_admissions'].apply(clean_recurring_admissions)

# Optional: Check result
#print(df['does_patient_have_recurring_admissions'].unique())

def clean_leukocyte_count(value):
    try:
        return float(value)
    except:
        return float(0)

# Apply it to the 'leukocyte_count_in_urine' column
df['leukocyte_count_in_urine'] = df['leukocyte_count_in_urine'].apply(clean_leukocyte_count)

# Optional: Check result
#print(df['leukocyte_count_in_urine'].unique())

def clean_leukocyte_esterase(value):
    if value in ['POZITIF', 'POZITF', 1]:
        return 1
    elif value in ['BAKILMAMIS', 'BAKILMAMIS']:
        return 0
    elif value in ['NEGATIF', 0]:
        return 0
    else:
        return value 

df['leukocyte_esterase_positive_in_urine'] = df['leukocyte_esterase_positive_in_urine'].apply(clean_leukocyte_esterase)


def clean_nitrite_positive(value):
    if value in ['POZITIF', 1]:
        return 1
    elif value in ['NEGATIF', 'NEGTAIF', 0, 9, 'BAKILMAMIS', 'BAKILMAMIS']:
        return 0
    else:
        return value 

# Apply it to the 'nitrite_positive_in_urine' column
df['nitrite_positive_in_urine'] = df['nitrite_positive_in_urine'].apply(clean_nitrite_positive)

# Optional: Check result
#print(df['nitrite_positive_in_urine'].unique())

def clean_erythrocyte_positive(value):
    if value == 'POZITIF':
        return 1  # Positive
    elif value == 'NEGATIF':
        return 0  # Negative
    elif value in ['BAKILMAMIS', 'BAKILMAMIS']:
        return 0
    try:
        # Convert to integer if possible
        num_value = flaot(value)
        if num_value < 5.0:
            return 0
        elif num_value >= 5:
            return 1
    except:
        # If conversion fails (non-numeric values)
        return value

# Apply it to the 'erythrocyte_positive_in_urine' column
df['erythrocyte_positive_in_urine'] = df['erythrocyte_positive_in_urine'].apply(clean_erythrocyte_positive)

# Optional: Check result
#print(df['erythrocyte_positive_in_urine'].unique())

def clean_bacteria_presence(value):
    if value in ['BAKILMAMIS', 'BAKILMAMIS', 'BAKILMAMIŞ']:
        return 0  # Invalid values, convert to NaN
    return value  # Keep numeric values and NaN unchanged

# Apply it to the 'bacteria_presence_in_urine' column
df['bacteria_presence_in_urine'] = df['bacteria_presence_in_urine'].apply(clean_bacteria_presence)

# Optional: Check result
#print(df['bacteria_presence_in_urine'].unique())

def clean_protein_in_urine(value):
    if value == 'POZITIF':
        return 1  # Positive
    elif value in ['NEGATIF', 'NEGAITF', 'NEGTAIF']:
        return 0  # Negative
    elif value in ['BAKILMAMIS', 'BAKILMAMIS']:
        return 0  
    elif value == 1:
        return 1  # Keep as is
    elif value == 0:
        return 0  # Keep as is
    return value  # For any other unhandled values

# Apply it to the 'protein_in_urine' column
df['protein_in_urine'] = df['protein_in_urine'].apply(clean_protein_in_urine)

# Optional: Check result
#print(df['protein_in_urine'].unique())

# Replace 'BAKILMAMIS', 'BAKILMAMIS' with NaN
df['USG'] = df['USG'].replace(['BAKILMAMIS', 'BAKILMAMIS'], np.nan)

# One-hot encode the cleaned USG column
usg_dummies = pd.get_dummies(df['USG'], prefix='USG')

# Concatenate one-hot encoded columns back to the original dataframe
df = pd.concat([df, usg_dummies], axis=1)
df[usg_dummies.columns] = df[usg_dummies.columns].astype(int)

#print(df.filter(like='USG_').head())


def clean_wbc(value):
    try:
        return float(value)
    except:
        # If it can't be converted to float (like 'BAKILMAMIS'), return NaN
        return np.nan

# Apply the function to the 'WBC' column
df['WBC'] = df['WBC'].apply(clean_wbc)

# Optional: Check result
#print(df['WBC'].unique())

def clean_hgb(value):
    try:
        return float(value)
    except:
        # If it can't be converted to float (like 'BAKILMAMIS'), return NaN
        return np.nan

# Apply the function to the 'HGB' column
df['HGB'] = df['HGB'].apply(clean_hgb)

# Optional: Check result
#print(df['HGB'].unique())

def clean_neu(value):
    try:
        return float(value)
    except:
        # If it can't be converted to float (like 'BAKILMAMIS'), return NaN
        return np.nan

# Apply the function to the 'NEU#' column
df['NEU#'] = df['NEU#'].apply(clean_neu)

# Optional: Check result
#print(df['NEU#'].unique())

def clean_lym(value):
    # Attempt to convert value to float (this will handle decimals correctly)
    try:
        return float(value)
    except:
        # If it can't be converted to float (like 'BAKILMAMIS'), return NaN
        return np.nan

# Apply the function to the 'LYM#' column
df['LYM#'] = df['LYM#'].apply(clean_lym)

# Optional: Check result
#print(df['LYM#'].unique())

def clean_urea(value):
    # Attempt to convert value to float (this will handle decimals correctly)
    try:
        return float(value)
    except:
        # If it can't be converted to float (like 'BAKILMAMIS'), return NaN
        return np.nan

# Apply the function to the 'urea' column
df['urea'] = df['urea'].apply(clean_urea)

# Optional: Check result
#print(df['urea'].unique())

def clean_bun(value):
    # Attempt to convert value to float (this will handle decimals correctly)
    try:
        return float(value)
    except:
        # If it can't be converted to float (like 'BAKILMAMIS'), return NaN
        return np.nan

# Apply the function to the 'urea' column
df['BUN'] = df['BUN'].apply(clean_bun)

# Optional: Check result
#print(df['BUN'].unique())

def clean_creatinine(value):
    # Attempt to convert value to float (this will handle decimals correctly)
    try:
        return float(value)
    except:
        # If it can't be converted to float (like 'BAKILMAMIS'), return NaN
        return np.nan

# Apply the function to the 'urea' column
df['creatinine'] = df['creatinine'].apply(clean_creatinine)

# Optional: Check result
#-print(df['creatinine'].unique())

def clean_crp(value):
    # Attempt to convert value to float (this will handle decimals correctly)
    try:
        return float(value)
    except:
        # If it can't be converted to float (like 'BAKILMAMIS'), return NaN
        return np.nan

# Apply the function to the 'urea' column
df['CRP'] = df['CRP'].apply(clean_crp)

# Optional: Check result
#print(df['CRP'].unique())


def clean_procalcitonin(value):
    # Attempt to convert value to float (this will handle decimals correctly)
    try:
        return float(value)
    except:
        # If it can't be converted to float (like 'BAKILMAMIS'), return NaN
        return np.nan

# Apply the function to the 'urea' column
df['procalcitonin'] = df['procalcitonin'].apply(clean_procalcitonin)

# Optional: Check result
#print(df['procalcitonin'].unique())

# Step 1: Normalize text
df['pathogen_growing_in_urine'] = df['pathogen_growing_in_urine'].astype(str).str.upper().str.strip()

# Step 2: Clean entries
df['pathogen_growing_in_urine'] = df['pathogen_growing_in_urine'].replace({
    'UREME OLMADI': 'NO_GROWTH',
    'BAKILMAMIS': np.nan,
    '4 TIP BAKTERI UREMESI VAR. TEKRARI ONERILIR': np.nan 
})

# Step 3: Create binary flag column for any pathogen growth
df['pathogen_growth'] = df['pathogen_growing_in_urine'].apply(lambda x: 0 if x == 'NO_GROWTH' else (1 if pd.notnull(x) else np.nan))

# Step 4: One-hot encode actual pathogen types (excluding 'NO_GROWTH')
pathogen_dummies = pd.get_dummies(df['pathogen_growing_in_urine'], prefix='pathogen')
pathogen_dummies = pathogen_dummies.drop(columns=[col for col in pathogen_dummies.columns if 'NO_GROWTH' in col])

# Step 5: Concatenate everything
df = pd.concat([df, pathogen_dummies], axis=1)
df[pathogen_dummies.columns] = df[pathogen_dummies.columns].astype(int)

def clean_esbl(value):
    if value in [1, 'POZITIF', '+']:
        return 1
    if value in [0,'NEGATIF','-', np.nan]: 
        return 0
    else:
        return value 

# Apply the function to the 'esbl_positive' column
df['esbl_positive'] = df['esbl_positive'].apply(clean_esbl)

# Optional: Check the cleaned column
#print(df['esbl_positive'].head())



antibiotics = [
    'sefazolin', 'seftazidim', 'gentamisin', 'tmp_smx', 'sefiksim',
    'ertapenem', 'pip_tazo', 'amoxicillin_clavulanic_acid', 'seftriakson',
    'sefuroksim_aksetil', 'sefuroksim', 'ampisilin', 'nitrofurantoin',
    'siprofloksasin', 'fosfomisin', 'sefepim', 'amikasin', 'sefoksitin',
    'meropenem', 'sefotaksim'
]

# Standardized mapping
antibiotic_mapping = {
    'DUYARLI': 0, 'S': 0,
    'R': 1, 'DIRENCLI': 1, 'DRENCLI': 1, 'DIENCLI': 1,
    'AZ DUYARLI': 1, 'A': 1, 'D': np.nan,
    'BAKILMAMIS': np.nan, 'BAKILMAMIS': np.nan, 'bAKILMAMIS': np.nan,
    'UREME OLMADI': np.nan, '-': np.nan, 'BAKIMAMIS': np.nan, 'NAN': np.nan, '4': np.nan
}

# Apply cleaning to all antibiotic columns
for col in antibiotics:
    df[col] = df[col].astype(str).str.strip().str.upper()
    df[col] = df[col].replace(antibiotic_mapping)



# Print unique values for each column
for column in df.columns:
    unique_vals = df[column].unique()
    print(f"{column} ({len(unique_vals)} unique):")
    print(unique_vals)
    print("-" * 50)

df = df.drop(columns=['pathogen_growing_in_urine', 'USG', 'underlying_risk_factor'])

df.to_excel('final_dataframe.xlsx', index=False)