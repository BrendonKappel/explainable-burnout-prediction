import math
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "burnout_model.pkl"

REDEPLOYMENT_COLUMN = "Have you been redeployed to a different area, hospital, or specialty?"
DESTINATION_COLUMN = "If so, to which specialty or area have you been redeployed?"
COVID_COLUMN = "Have you treated a COVID-19 positive patient in the past week?"
DAYS_COLUMN = "How many days did you spend at work in the last 7 days?"
SHIFT_COLUMN = "What was the average duration of your shift?"
TRAINING_COLUMN = (
    "If redeployed: how would you describe the training you received in preparation for redeployment?"
)
SUPPORT_COLUMN = "If redeployed: how would you describe the support you received during redeployment?"
SHARED_COLUMNS = [
    "gender",
    "role",
    "Country",
    "base",
    REDEPLOYMENT_COLUMN,
    DESTINATION_COLUMN,
    COVID_COLUMN,
    DAYS_COLUMN,
    SHIFT_COLUMN,
    TRAINING_COLUMN,
    SUPPORT_COLUMN,
]

SAQ_COLUMNS = [f"SAQ.{item}" for item in range(1, 37) if item != 8]
FEATURE_COLUMNS = SHARED_COLUMNS + SAQ_COLUMNS
CATEGORICAL_COLUMNS = [column for column in SHARED_COLUMNS if column != DAYS_COLUMN]

MISSING_MARKERS = ["#VALUE!", "#DIV/0!", "NOT AVAILABLE", "NOT STATED"]
NO_DESTINATION = ["N/A", "NA", "NOT APPLICABLE", "NOT REDEPLOYED", "NOT DEPLOYED"]

SHIFT_CATEGORIES = {
    "LESS THAN 4 HOURS": "MENOS DE 4 HORAS",
    "4 TO LESS THAN 8 HOURS": "4 A MENOS DE 8 HORAS",
    "8 TO LESS THAN 12 HOURS": "8 A MENOS DE 12 HORAS",
    "12 TO LESS THAN 18 HOURS": "12 A MENOS DE 18 HORAS",
    "18 TO LESS THAN 24 HOURS": "18 A 24 HORAS",
    "MORE THAN 24 HOURS": "MAIS DE 24 HORAS",
}

base_groups = {
    "CLINICA_MEDICA": ["MEDICINE", "CARDIOLOGY"],
    "CIRURGICA": ["SURGERY", "THEATRE"],
    "ATENDIMENTO_AGUDO": ["EM", "ITU", "ANAESTHESIA", "TRAUMA", "PRE-HOSPITAL EM"],
    "OUTRAS_ESPECIALIDADES": ["OPD", "GP", "DIALYSIS", "NURSERY", "NEUROSCIENCE"],
    "NAO_ASSISTENCIAL": ["NON-CLINICAL"],
    "NAO_ESPECIFICADO": ['SPECIALTY', 'WARD', 'WARDOPD', 'OTHER', 'STUDENT', 'RESEARCH'],
}

destination_groups = {
    "ATENDIMENTO_CRITICO": [
        "INTENSIVE CARE", "INTENSIVE CARE MEDICINE", "EMERGENCY DEPARTMENT",
        "EMERGENCY MEDICINE", "ANAESTHESIA", "OVERFLOW ITU",
        "PRONING TEAM", "PRONING TEAM ON CRITICAL CARE",
        "INTENSIVE CARE AND ADULT COVID WARD",
        "EMERGENCY DEPARTMENT - MINOR INJURIES",
    ],
    "ATENDIMENTO_COVID": [
        "WARD TEAM TAKING CARE OF COVID-19 SUSPECTS / PATIENTS",
        "COVID WARD", "COVID COHORT WARD", "COVID19 WARD",
        "A SPECIFIED WARD FOR COVID-19 PATIENTS", "WORKING WITH COVID-19 PATIENTS",
    ],
    "HOSPITALAR_GERAL": [
        "GENERAL MEDICINE", "GENERAL MEDICINE (NOT MAINLY COVID-19 CASES)",
        "SURGERY", "INPATIENT NON-COVID SERVICES", "MATERNITY",
        "NON-COVID INPATIENT", "OTHER WARDS", "PAEDIATRICS", "RESPIRATORY",
        "OTHERS: PHARMACY", "RESPIRATORY MEDICINE", "EYES", "TRAUMA SURGERY",
        "STROKE", "PPE HELPER ON RESPIRATORY AND ONCOLOGY WARDS",
        "SURGICAL WARD HAS BEEN TRANSFORMED INTO AN OBSERVATION WARD",
        "AUDIOLOGY NEWBORN HEARING SCREENING", "LABOUR WARD",
        "INFECTION & IMMUNITY LAB", "ORTHOPAEDIC", "ENDOSCOPY",
        "WARD BASED CARE MIXED SPECIALITY", "REDEPLOYED TO HELP IN HOSPITALS",
        "DIALYSIS", "DIALYSIS STATION", "MENTAL HEALTH", "OTHER PSYCHIATRY WARD",
        "INTERNAL MEDICINE", "INFECTIOUS DISEASES", "AMU", "ONCOLOGY",
        "STROKE, NEUROLOGY AND NEUROSURGERY",
        "FROM HH TO CXH CHEMOTHERAPY DAY UNIT", "ACUTE MEDICAL WARD", "CARDIOLOGY",
        "PSYCHIATRY AND DETOX", "OTHERS: DIFFERENT DEPARTMENT IN LAB MEDICINE",
    ],
    "COMUNITARIO": [
        "FOREIGN WORKER DORMITORY",
        "COMMUNITY CARE FACILITY / COMMUNITY ISOLATION FACILITY / COMMUNITY SCREENING",
        "GENERAL PRACTICE / PRIMARY CARE", "OTHER FOREIGN WORKER HOUSING FACILITY",
        "PRIMARY CARE", "NURSING HOME",
    ],
    "NAO_ASSISTENCIAL": [
        "ADMIN", "OTHER MANAGEMENT", "MANAGEMENT", "CLERICAL",
        "COVID RESEARCH", "COVID RESEARCH TRIALS", "RESEARCH",
        "DIFFERENT RESEARCH SPECIALITY",
        "RESEARCH DEPARTMENT OF THE NEAREST ACUTE TRUST.",
        "COVID INCIDENT CONTROL ROOM", "COVID-19 TASK FORCE",
        "OTHERS: DISEASE OUTBREAK TASK FORCE", "OTHERS: TRAINING DEPT",
        "OTHERS: COVID RESEARCH",
        "COVID TRIALS AND I AM SHIELDING, SO WORKING FROM HOME",
    ],
    "NAO_ESPECIFICADO": [
        "COVID",
        'NOT LISTED',
        'MIXTURE OF GENERAL PRACTICE AND EVERY FOURTH WEEK WORKING WITH THE HOSPITAL PAEDIATRIC TEAM ON '
        'AN ADULT COVID WARD',
        'THE NIGHTINGALE',
        'WORKING FROM HOME AS PREGNANT (30 WEEKS AT TIME OF RCOG GUIDANCE FOR PREGNANCY)',
        'COVER 8TH FLOOR WARDS AND ED ADMISSIONS',
        'SAME SPECIALTY, DIFFERENT LOCATION - MOVED TO SITE BASED RATHER THAN DEPT BASED',
        'EVERY WHERE, I ASKED TO REPORT MYSELF TO SIDE MANAGER AND WILL BE SEND TO NEW AREA EVERYDAY.',
        'DISCHARGE',
        'COVID-19 STAFF SCREENING, COVID-19 TRUST RESEARCH TEAM',
        'PATIENT SCREENING',
        'MASK FIT TESTING',
        'SCREENING PATIENTS AT ENTRY AND SWABBING STAFF FOR COVID, ALSO THE HAEMATOLOGY WARD',
        'INFECTION CONTROL',
        'PUNKT POBORU WYMAZOW COVID- DORAZNIE',
        'JESTESMY CIAGLE PRZERZUCANI NA ROZNE ODDZIAŁY W CAŁYM SZPITALU, W ZALEZNOSCI OD BRAKOW PERSONELU '
        'W DANYM DNIU W DANEJ CZESCI SZPITALA',
        'QUARANTINE',
        'IZOLATORIUM ODDZ. BUFOROWY',
    ],
}

quality_groups = {
    "GOOD": "POSITIVA", "VERY GOOD": "POSITIVA",
    "NEUTRAL": "NEUTRA", "POOR": "NEGATIVA", "VERY POOR": "NEGATIVA",
}
training_groups = {
    **quality_groups,
    "NO TRAINING RECEIVED": "SEM_TREINAMENTO",
    "SLIGHTLY AGREE": "RESPOSTA_INCOMPATIVEL",
    "STRONGLY DISAGREE": "RESPOSTA_INCOMPATIVEL",
}
support_groups = {
    **quality_groups,
    "NO SUPPORT RECEIVED": "SEM_SUPORTE",
    "NO TRAINING RECEIVED": "RESPOSTA_INCOMPATIVEL",
    "SLIGHTLY AGREE": "RESPOSTA_INCOMPATIVEL",
    "STRONGLY DISAGREE": "RESPOSTA_INCOMPATIVEL",
}


def normalize_text(value):
    text = unicodedata.normalize("NFKD", value)
    text = "".join(character for character in text if not unicodedata.combining(character))
    return text.upper().strip()


def group_shift_duration(value):
    if pd.isna(value):
        return pd.NA
    if value in SHIFT_CATEGORIES:
        return SHIFT_CATEGORIES[value]
    if value in SHIFT_CATEGORIES.values():
        return value

    hours = float(value)
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError("A duração do turno deve ser um número finito maior que zero.")
    if hours < 4:
        return "MENOS DE 4 HORAS"
    if hours < 8:
        return "4 A MENOS DE 8 HORAS"
    if hours < 12:
        return "8 A MENOS DE 12 HORAS"
    if hours < 18:
        return "12 A MENOS DE 18 HORAS"
    if hours <= 24:
        return "18 A 24 HORAS"
    return "MAIS DE 24 HORAS"


def flatten_groups(groups):
    return {response: group for group, responses in groups.items() for response in responses}


GROUP_MAPPINGS = {
    "base": flatten_groups(base_groups),
    DESTINATION_COLUMN: flatten_groups(destination_groups),
    TRAINING_COLUMN: training_groups,
    SUPPORT_COLUMN: support_groups,
}


def prepare_inputs(raw):
    missing_columns = [column for column in FEATURE_COLUMNS if column not in raw.columns]
    if missing_columns:
        raise ValueError("Colunas ausentes no CSV: " + ", ".join(missing_columns))
    if raw.empty:
        raise ValueError("O CSV precisa conter pelo menos um registro.")

    features = raw[FEATURE_COLUMNS].copy()
    features[CATEGORICAL_COLUMNS] = features[CATEGORICAL_COLUMNS].astype("string")
    for column in features.select_dtypes(include=["object", "string"]).columns:
        values = features[column].astype("string")
        features[column] = values.map(normalize_text, na_action="ignore").astype("string")
    features = features.replace([*MISSING_MARKERS, ""], pd.NA)

    features[COVID_COLUMN] = features[COVID_COLUMN].replace({"I DON'T KNOW": "DON'T KNOW"})
    features[SHIFT_COLUMN] = features[SHIFT_COLUMN].map(group_shift_duration)
    features.loc[features["role"].eq("STAFF_CAT").fillna(False), "role"] = pd.NA

    not_redeployed = features[REDEPLOYMENT_COLUMN].eq("NO").fillna(False)
    features.loc[features[DESTINATION_COLUMN].isin(NO_DESTINATION), DESTINATION_COLUMN] = pd.NA
    features.loc[not_redeployed, [DESTINATION_COLUMN, TRAINING_COLUMN, SUPPORT_COLUMN]] = "NAO_SE_APLICA"

    for column, mapping in GROUP_MAPPINGS.items():
        values = features[column].astype("string")
        unmapped = values.notna() & ~values.isin([*mapping, "NAO_SE_APLICA"])
        if unmapped.any():
            print(f"Respostas não mapeadas em {column}: {unmapped.sum()}.")
        features[column] = values.replace(mapping)

    features[CATEGORICAL_COLUMNS] = features[CATEGORICAL_COLUMNS].astype("string")
    days = pd.to_numeric(features[DAYS_COLUMN])
    reported_days = days.dropna()
    if not reported_days.between(0, 7).all() or not reported_days.mod(1).eq(0).all():
        raise ValueError("A quantidade de dias trabalhados deve ser um inteiro entre 0 e 7.")
    features[DAYS_COLUMN] = days.astype("Int64")
    for column in SAQ_COLUMNS:
        values = pd.to_numeric(features[column])
        if not values.dropna().isin([0, 1]).all():
            raise ValueError(f"{column} deve conter 0, 1 ou um valor ausente.")
        features[column] = values.astype("Int64")
    return features


def build_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(
            missing_values=pd.NA, strategy="constant", fill_value="NAO INFORMADO",
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    binary = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, [DAYS_COLUMN]),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
        ("binary", binary, SAQ_COLUMNS),
    ])
