import streamlit as st
import pandas as pd
import joblib

# Load CSV
raw_data = pd.read_csv("data.csv", sep=';')

# Define target and feature types
target_col = 'Target'
categorical_features = [
    'Marital status', 'Application mode', 'Course', 'Daytime/evening attendance\t',
    'Previous qualification', 'Nacionality', 'Mother\'s qualification', 'Father\'s qualification',
    'Mother\'s occupation', 'Father\'s occupation', 'Displaced', 'Educational special needs',
    'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
]
numeric_features = [col for col in raw_data.columns if col not in categorical_features + [target_col]]

# Feature descriptions mapping
feature_descriptions = {
    'Marital status': {
        1: 'Single',
        2: 'Married',
        3: 'Widower',
        4: 'Divorced',
        5: 'Facto union',
        6: 'Legally separated'
    },
    'Application mode': {
        1: '1st phase - general contingent',
        2: 'Ordinance No. 612/93',
        5: '1st phase - special contingent (Azores Island)',
        7: 'Holders of other higher courses',
        10: 'Ordinance No. 854-B/99',
        15: 'International student (bachelor)',
        16: '1st phase - special contingent (Madeira Island)',
        17: '2nd phase - general contingent',
        18: '3rd phase - general contingent',
        26: 'Ordinance No. 533-A/99, item b2) (Different Plan)',
        27: 'Ordinance No. 533-A/99, item b3 (Other Institution)',
        39: 'Over 23 years old',
        42: 'Transfer',
        43: 'Change of course',
        44: 'Technological specialization diploma holders',
        51: 'Change of institution/course',
        53: 'Short cycle diploma holders',
        57: 'Change of institution/course (International)'
    },
    'Course': {
        33: 'Biofuel Production Technologies',
        171: 'Animation and Multimedia Design',
        8014: 'Social Service (evening attendance)',
        9003: 'Agronomy',
        9070: 'Communication Design',
        9085: 'Veterinary Nursing',
        9119: 'Informatics Engineering',
        9130: 'Equinculture',
        9147: 'Management',
        9238: 'Social Service',
        9254: 'Tourism',
        9500: 'Nursing',
        9556: 'Oral Hygiene',
        9670: 'Advertising and Marketing Management',
        9773: 'Journalism and Communication',
        9853: 'Basic Education',
        9991: 'Management (evening attendance)'
    },
    'Daytime/evening attendance\t': {
        1: 'Daytime',
        0: 'Evening'
    },
    'Previous qualification': {
        1: 'Secondary education',
        2: 'Higher education - bachelor\'s degree',
        3: 'Higher education - degree',
        4: 'Higher education - master\'s',
        5: 'Higher education - doctorate',
        6: 'Frequency of higher education',
        9: '12th year of schooling - not completed',
        10: '11th year of schooling - not completed',
        12: 'Other - 11th year of schooling',
        14: '10th year of schooling',
        15: '10th year of schooling - not completed',
        19: 'Basic education 3rd cycle (9th/10th/11th year) or equiv.',
        38: 'Basic education 2nd cycle (6th/7th/8th year) or equiv.',
        39: 'Technological specialization course',
        40: 'Higher education - degree (1st cycle)',
        42: 'Professional higher technical course',
        43: 'Higher education - master (2nd cycle)'
    },
    'Nacionality': {
        1: 'Portuguese', 2: 'German', 6: 'Spanish', 11: 'Italian', 13: 'Dutch', 14: 'English',
        17: 'Lithuanian', 21: 'Angolan', 22: 'Cape Verdean', 24: 'Guinean', 25: 'Mozambican',
        26: 'Santomean', 32: 'Turkish', 41: 'Brazilian', 62: 'Romanian', 100: 'Moldova (Republic of)',
        101: 'Mexican', 103: 'Ukrainian', 105: 'Russian', 108: 'Cuban', 109: 'Colombian'
    },
    'Mother\'s qualification': {
        1: 'Secondary Education - 12th Year of Schooling or Eq.',
        2: 'Higher Education - Bachelor\'s Degree',
        3: 'Higher Education - Degree',
        4: 'Higher Education - Master\'s',
        5: 'Higher Education - Doctorate',
        6: 'Frequency of Higher Education',
        9: '12th Year of Schooling - Not Completed',
        10: '11th Year of Schooling - Not Completed',
        11: '7th Year (Old)',
        12: 'Other - 11th Year of Schooling',
        14: '10th Year of Schooling',
        18: 'General commerce course',
        19: 'Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.',
        22: 'Technical-professional course',
        26: '7th year of schooling',
        27: '2nd cycle of the general high school course',
        29: '9th Year of Schooling - Not Completed',
        30: '8th year of schooling',
        34: 'Unknown',
        35: 'Can\'t read or write',
        36: 'Can read without having a 4th year of schooling',
        37: 'Basic education 1st cycle (4th/5th year) or equiv.',
        38: 'Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.',
        39: 'Technological specialization course',
        40: 'Higher education - degree (1st cycle)',
        41: 'Specialized higher studies course',
        42: 'Professional higher technical course',
        43: 'Higher Education - Master (2nd cycle)',
        44: 'Higher Education - Doctorate (3rd cycle)'
    },
    'Father\'s qualification': {
        1: 'Secondary Education - 12th Year of Schooling or Eq.',
        2: 'Higher Education - Bachelor\'s Degree',
        3: 'Higher Education - Degree',
        4: 'Higher Education - Master\'s',
        5: 'Higher Education - Doctorate',
        6: 'Frequency of Higher Education',
        9: '12th Year of Schooling - Not Completed',
        10: '11th Year of Schooling - Not Completed',
        11: '7th Year (Old)',
        12: 'Other - 11th Year of Schooling',
        13: '2nd year complementary high school course',
        14: '10th Year of Schooling',
        18: 'General commerce course',
        19: 'Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.',
        20: 'Complementary High School Course',
        22: 'Technical-professional course',
        25: 'Complementary High School Course - not concluded',
        26: '7th year of schooling',
        27: '2nd cycle of the general high school course',
        29: '9th Year of Schooling - Not Completed',
        30: '8th year of schooling',
        31: 'General Course of Administration and Commerce',
        33: 'Supplementary Accounting and Administration',
        34: 'Unknown',
        35: 'Can\'t read or write',
        36: 'Can read without having a 4th year of schooling',
        37: 'Basic education 1st cycle (4th/5th year) or equiv.',
        38: 'Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.',
        39: 'Technological specialization course',
        40: 'Higher education - degree (1st cycle)',
        41: 'Specialized higher studies course',
        42: 'Professional higher technical course',
        43: 'Higher Education - Master (2nd cycle)',
        44: 'Higher Education - Doctorate (3rd cycle)'
    },
    'Displaced': {
        1: 'Yes',
        0: 'No'
    },
    'Educational special needs': {
        1: 'Yes',
        0: 'No'
    },
    'Debtor': {
        1: 'Yes',
        0: 'No'
    },
    'Tuition fees up to date': {
        1: 'Yes',
        0: 'No'
    },
    'Gender': {
        1: 'Male',
        0: 'Female'
    },
    'Scholarship holder': {
        1: 'Yes',
        0: 'No'
    },
    'International': {
        1: 'Yes',
        0: 'No'
    },
        # Mother's occupation
    "Mother's occupation": {
        0: 'Student', 1: 'Legislative/Executive Power', 2: 'Intellectual Activities',
        3: 'Intermediate Technicians', 4: 'Administrative staff', 5: 'Services/Security/Sellers',
        6: 'Farmers and Skilled Agriculture', 7: 'Skilled Industry/Construction',
        8: 'Machine Operators', 9: 'Unskilled Workers', 10: 'Armed Forces',
        90: 'Other Situation', 99: '(blank)', 122: 'Health professionals', 123: 'Teachers',
        125: 'ICT Specialists', 131: 'Science/Engineering Technicians',
        132: 'Health Technicians', 134: 'Legal/Social/Sports/Cultural Technicians',
        141: 'Secretaries/Data Entry', 143: 'Accounting/Finance/Registry Operators',
        144: 'Other Admin Support', 151: 'Service Workers', 152: 'Sellers',
        153: 'Personal Care Workers', 171: 'Construction Workers',
        173: 'Printing/Instrument/Jewelry/Craft', 175: 'Food/Wood/Clothing Craftsmen',
        191: 'Cleaning Workers', 192: 'Unskilled Agriculture',
        193: 'Unskilled Industry/Construction/Transport', 194: 'Meal Prep Assistants'
    },
    # Father's occupation
    "Father's occupation": {
        0: 'Student', 1: 'Legislative/Executive Power', 2: 'Intellectual Activities',
        3: 'Intermediate Technicians', 4: 'Administrative staff', 5: 'Services/Security/Sellers',
        6: 'Farmers and Skilled Agriculture', 7: 'Skilled Industry/Construction',
        8: 'Machine Operators', 9: 'Unskilled Workers', 10: 'Armed Forces', 90: 'Other Situation',
        99: '(blank)', 101: 'Armed Forces Officers', 102: 'Armed Forces Sergeants',
        103: 'Other Armed Forces Personnel', 112: 'Admin/Commercial Directors',
        114: 'Hotel/Catering Directors', 121: 'Science/Engineering Specialists',
        122: 'Health professionals', 123: 'Teachers', 124: 'Finance/Admin/Public Relations',
        131: 'Science/Engineering Technicians', 132: 'Health Technicians',
        134: 'Legal/Social/Sports/Cultural Technicians', 135: 'ICT Technicians',
        141: 'Secretaries/Data Entry', 143: 'Accounting/Finance/Registry Operators',
        144: 'Other Admin Support', 151: 'Service Workers', 152: 'Sellers',
        153: 'Personal Care Workers', 154: 'Protection/Security',
        161: 'Market-Oriented Farmers', 163: 'Subsistence Farmers/Fishermen',
        171: 'Construction Workers', 172: 'Metallurgy/Metalworking',
        174: 'Electricians/Electronics', 175: 'Food/Wood/Clothing Craftsmen',
        181: 'Fixed Machine Operators', 182: 'Assembly Workers',
        183: 'Vehicle/Mobile Equipment Operators', 192: 'Unskilled Agriculture',
        193: 'Unskilled Industry/Construction/Transport', 194: 'Meal Prep Assistants',
        195: 'Street Vendors/Services'
    }
}

# Load artifacts
model = joblib.load('best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
target_map = joblib.load('target_map.pkl')
inv_target = {v: k for k, v in target_map.items()}

st.title('Student Status Prediction')
st.write("Input features to predict: Dropout, Enrolled, or Graduate.")

# Build inputs
df_defaults = raw_data.drop(columns=[target_col])
input_dict = {}

for col in df_defaults.columns:
    if col in numeric_features:
        input_dict[col] = st.number_input(col, value=float(df_defaults[col].mean()))
    else:
        # Get unique values and sort them
        opts = sorted(df_defaults[col].unique())
        
        # Create display labels using feature descriptions if available
        if col in feature_descriptions:
            display_opts = [f"{code} - {feature_descriptions[col].get(code, 'Unknown')}" for code in opts]
        else:
            display_opts = [str(code) for code in opts]
        
        # Create selection box with display labels
        selected_display = st.selectbox(col, display_opts)
        
        # Extract the actual value from the selection
        if col in feature_descriptions:
            # Find the code that matches the selected description
            selected_code = next(
                (code for code in opts 
                 if selected_display.startswith(f"{code} -")),
                opts[0]  # default to first option if not found
            )
        else:
            selected_code = int(selected_display.split()[0])
        
        input_dict[col] = selected_code

input_df = pd.DataFrame([input_dict])
X_prep = preprocessor.transform(input_df)
pred_idx = model.predict(X_prep)[0]
probs = model.predict_proba(X_prep)[0]

st.subheader('Prediction')
st.write(f"**{inv_target[pred_idx]}**")
st.write("Class Probabilities:")
st.table(pd.DataFrame({'Class': list(inv_target.values()), 'Probability': probs}))