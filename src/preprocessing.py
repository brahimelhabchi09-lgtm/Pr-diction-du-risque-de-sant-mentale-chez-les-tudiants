from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


numeric_columns = [
    'Age', 'Year_of_Study', 'Sleep_Hours', 'Study_Hours_Per_Week', 'GPA',
    'Social_Media_Hours', 'Screen_Time_Hours', 'Financial_Stress',
    'Family_Support', 'Attendance_Rate',
]
categorical_columns = [
    'Gender', 'Physical_Activity', 'Part_Time_Job', 'Diet_Quality',
    'Previous_Mental_Health_History',
]

numeric_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
])
categorical_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_columns),
    ('categorical', categorical_pipeline, categorical_columns),
])


def prepare_preprocessing():
    print(preprocessor)


if __name__ == '__main__':
    prepare_preprocessing()
