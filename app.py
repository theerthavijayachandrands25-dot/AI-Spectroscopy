from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import savgol_filter

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "outputs" / "processed_data"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "feature_selection"
    / "extra_trees_top_1000.pkl"
)


def find_project_file(filename):
    """Find a data file in the preferred folder or elsewhere in the project."""
    preferred_path = DATA_DIR / filename

    if preferred_path.exists():
        return preferred_path

    matches = list(BASE_DIR.rglob(filename))
    return matches[0] if matches else None


X_PATH = find_project_file("X_features.csv")
Y_PATH = find_project_file("Y_labels.csv")
METADATA_PATH = find_project_file("molecule_metadata.csv")


# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI-Based IR Spectroscopy Analyzer",
    page_icon="🧪",
    layout="wide"
)


# =========================================================
# 3. FUNCTIONAL GROUP SMARTS
# =========================================================
# These SMARTS patterns provide a structural cross-check when a
# molecule's SMILES is available. They do NOT override the ML model.

SMARTS_PATTERNS = {
    "Alcohol": "[CX4][OX2H]",
    "Ether": "[OD2]([#6])[#6]",
    "Amine": "[NX3;!$(NC=O)]",
    "Alkyne": "[CX2]#[CX2]",
    "Nitrile": "[C]#N",
    "Aldehyde": "[CX3H1](=O)[#6]",
    "Amide": "[NX3][CX3](=[OX1])[#6]",
    "Ketone": "[#6][CX3](=O)[#6]",
    "Ester": "[CX3](=O)[OX2H0][#6]",
    "Imine": "[CX3]=[NX2]",
    "Alkene": "[CX3]=[CX3]",
    "Phenol": "[OX2H]c",
    "Amidine": "[CX3](=[NX2])[NX3]",
    "Oxime": "[CX3]=[NX2][OX2H]",
    "Acetal": "[CX4]([OX2])([OX2])",
    "Fluoro Compound": "[#6][F]"
}


def check_smiles_functional_groups(smiles, labels):
    """
    Return a structural check for labels that have a SMARTS pattern.
    'Not checked' means no matching pattern is defined for that label.
    """
    if not RDKIT_AVAILABLE or not str(smiles).strip():
        return {}

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return {str(label): "Invalid SMILES" for label in labels}

    checks = {}
    for label in labels:
        label = str(label)
        pattern = SMARTS_PATTERNS.get(label)

        if pattern is None:
            checks[label] = "Not checked"
            continue

        query = Chem.MolFromSmarts(pattern)
        if query is None:
            checks[label] = "Pattern error"
            continue

        checks[label] = (
            "Structure match"
            if molecule.HasSubstructMatch(query)
            else "No structure match"
        )

    return checks


# =========================================================
# 4. LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():
    if X_PATH is None:
        raise FileNotFoundError("X_features.csv was not found.")

    if Y_PATH is None:
        raise FileNotFoundError("Y_labels.csv was not found.")

    if METADATA_PATH is None:
        raise FileNotFoundError("molecule_metadata.csv was not found.")

    X = pd.read_csv(X_PATH)
    Y = pd.read_csv(Y_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    # Remove saved index columns from X.
    X = X.loc[:, ~X.columns.astype(str).str.startswith("Unnamed:")]

    # Convert spectral features to numeric values.
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isnull().any().any():
        raise ValueError(
            "X_features.csv contains missing or invalid values."
        )

    if X.shape[1] != 3501:
        raise ValueError(
            f"Expected 3501 spectral features, found {X.shape[1]}."
        )

    if not (len(X) == len(Y) == len(metadata)):
        raise ValueError(
            "The dataset CSV files have different row counts."
        )

    # Reset indices so the three files stay aligned by row.
    X = X.reset_index(drop=True)
    Y = Y.reset_index(drop=True)
    metadata = metadata.reset_index(drop=True)

    return X, Y, metadata


# =========================================================
# 5. LOAD EXTRA TREES MODEL
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Extra Trees model not found: {MODEL_PATH}"
        )

    saved_data = joblib.load(MODEL_PATH)

    required_keys = {
        "model",
        "labels",
        "thresholds",
        "selected_features"
    }
    missing_keys = required_keys.difference(saved_data.keys())

    if missing_keys:
        raise KeyError(
            "The saved model file is missing required items: "
            + ", ".join(sorted(missing_keys))
        )

    model = saved_data["model"]
    labels = list(saved_data["labels"])
    thresholds = np.asarray(saved_data["thresholds"], dtype=float)
    selected_features = list(saved_data["selected_features"])

    if len(selected_features) != 1000:
        raise ValueError(
            "The saved model must contain 1000 selected features."
        )

    if not hasattr(model, "estimators_"):
        raise ValueError(
            "The saved model does not expose estimators_. "
            "The prediction code expects a multi-output Extra Trees model."
        )

    if len(model.estimators_) != len(labels):
        raise ValueError(
            "The model estimator count does not match the labels."
        )

    if len(labels) != len(thresholds):
        raise ValueError(
            "The label count does not match the thresholds."
        )

    # Confirm each binary estimator expects 1000 features.
    for estimator in model.estimators_:
        if getattr(estimator, "n_features_in_", None) != 1000:
            raise ValueError(
                "At least one Extra Trees estimator does not expect "
                "1000 features."
            )

    if not np.all(np.isfinite(thresholds)):
        raise ValueError("Saved thresholds contain invalid values.")

    if np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError(
            "Saved thresholds must be between 0 and 1."
        )

    return saved_data


# =========================================================
# 6. FEATURE SELECTION AND PREDICTION
# =========================================================

def select_model_features(spectrum, feature_columns, selected_features):
    """
    Select the same 1000 features, in the saved order.

    Supports either saved feature names or saved integer column positions.
    """
    spectrum_df = pd.DataFrame(
        [np.asarray(spectrum, dtype=float)],
        columns=feature_columns
    )

    if all(isinstance(item, (int, np.integer)) for item in selected_features):
        if min(selected_features) < 0 or max(selected_features) >= len(feature_columns):
            raise ValueError(
                "A saved selected-feature index is outside the 3501 input features."
            )
        selected_spectrum = spectrum_df.iloc[:, selected_features]
    else:
        missing = [
            feature for feature in selected_features
            if feature not in spectrum_df.columns
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} selected model features are missing "
                "from X_features.csv. Check feature names and order."
            )
        selected_spectrum = spectrum_df.loc[:, selected_features]

    return selected_spectrum


def predict_functional_groups(spectrum, smiles=None, threshold_adjustment=0.0):
    """
    Predict functional groups using the existing saved model.

    threshold_adjustment is an optional experimental offset applied equally
    to all saved thresholds. Default 0.0 preserves the original thresholds.
    It should not be treated as calibrated unless validated on held-out data.
    """
    saved_data = load_model()
    X, _, _ = load_dataset()

    model = saved_data["model"]
    labels = list(saved_data["labels"])
    thresholds = np.asarray(saved_data["thresholds"], dtype=float)
    selected_features = list(saved_data["selected_features"])

    spectrum = np.asarray(spectrum, dtype=float).flatten()

    if len(spectrum) != 3501:
        raise ValueError(
            f"Expected 3501 spectral values, found {len(spectrum)}."
        )

    if not np.all(np.isfinite(spectrum)):
        raise ValueError("Spectrum contains invalid or non-finite values.")

    selected_spectrum = select_model_features(
        spectrum,
        X.columns,
        selected_features
    )

    # Optional exploratory adjustment. At 0.0, saved thresholds are unchanged.
    applied_thresholds = np.clip(
        thresholds + float(threshold_adjustment),
        0.0,
        1.0
    )

    results = []

    for i, estimator in enumerate(model.estimators_):
        probabilities = estimator.predict_proba(selected_spectrum)
        classes = list(estimator.classes_)

        if 1 in classes:
            positive_position = classes.index(1)
            probability = float(probabilities[0][positive_position])
        else:
            probability = 0.0

        threshold = float(applied_thresholds[i])
        prediction = probability >= threshold

        results.append({
            "Functional Group": str(labels[i]),
            "Probability (%)": round(probability * 100, 2),
            "Threshold (%)": round(threshold * 100, 2),
            "Prediction": "Detected" if prediction else "Not Detected"
        })

    results_df = pd.DataFrame(results)

    if smiles:
        structural_checks = check_smiles_functional_groups(smiles, labels)
        results_df["SMILES Structural Check"] = (
            results_df["Functional Group"].map(structural_checks)
        )

    return results_df


# =========================================================
# 7. PREPROCESS RAW IR SPECTRUM
# =========================================================

def preprocess_spectrum(values):
    """
    This preprocessing is retained from the supplied app.py.

    IMPORTANT: Use this for raw uploads only if it exactly matches the
    preprocessing used to create the training spectra. Otherwise, update
    this function to match the training pipeline.
    """
    values = np.asarray(values, dtype=float).flatten()

    if len(values) != 3501:
        raise ValueError(
            f"Expected 3501 values, received {len(values)}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("Spectrum contains invalid values.")

    # Savitzky-Golay requires an odd window length greater than polyorder.
    smoothed = savgol_filter(
        values,
        window_length=11,
        polyorder=3
    )

    maximum = np.max(np.abs(smoothed))

    if maximum == 0:
        raise ValueError("Spectrum contains only zeros.")

    return smoothed / maximum


# =========================================================
# 8. MOLECULAR STRUCTURE VISUALIZATION
# =========================================================

def draw_molecule(smiles, functional_groups):
    if not RDKIT_AVAILABLE:
        st.warning(
            "RDKit is not installed. Molecular structure display "
            "and SMILES validation are unavailable."
        )
        return

    molecule = Chem.MolFromSmiles(str(smiles))

    if molecule is None:
        st.error("Invalid SMILES structure.")
        return

    highlighted_atoms = set()

    for group in functional_groups:
        pattern = SMARTS_PATTERNS.get(str(group))

        if pattern is None:
            continue

        query = Chem.MolFromSmarts(pattern)
        if query is None:
            continue

        matches = molecule.GetSubstructMatches(query)
        for match in matches:
            highlighted_atoms.update(match)

    image = Draw.MolToImage(
        molecule,
        size=(650, 450),
        highlightAtoms=list(highlighted_atoms)
    )

    st.image(image, caption="Molecular structure")


# =========================================================
# 9. HELPER DISPLAY FUNCTIONS
# =========================================================

def show_prediction_results(results):
    detected = results[results["Prediction"] == "Detected"]

    st.subheader("AI Functional Group Predictions")

    col1, col2 = st.columns(2)
    col1.metric("Detected Groups", len(detected))
    col2.metric("Total Groups", len(results))

    if detected.empty:
        st.info("No functional groups crossed their prediction thresholds.")
    else:
        st.success(
            "Detected: "
            + ", ".join(detected["Functional Group"].tolist())
        )

    st.subheader("All Prediction Results")
    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    return detected


def show_actual_label_comparison(results, actual_labels):
    predicted_labels = results.loc[
        results["Prediction"] == "Detected",
        "Functional Group"
    ].tolist()

    true_positives = [
        group for group in predicted_labels if group in actual_labels
    ]
    false_positives = [
        group for group in predicted_labels if group not in actual_labels
    ]
    false_negatives = [
        group for group in actual_labels if group not in predicted_labels
    ]

    st.subheader("Comparison with Dataset Labels")

    st.caption(
        "This comparison uses Y_labels.csv as the reference. "
        "It is not an independent chemical verification."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Matching Predictions", len(true_positives))
    col2.metric("Predicted but Not in Labels", len(false_positives))
    col3.metric("Labels Not Predicted", len(false_negatives))

    if false_positives:
        st.warning(
            "Predicted groups not present in the dataset labels: "
            + ", ".join(false_positives)
        )

    if false_negatives:
        st.info(
            "Dataset labels not detected by the model: "
            + ", ".join(false_negatives)
        )


def show_threshold_control():
    st.sidebar.subheader("Prediction threshold")
    st.sidebar.caption(
        "The default 0.00 adjustment uses the thresholds saved with the model."
    )
    adjustment = st.sidebar.slider(
        "Experimental threshold adjustment",
        min_value=-0.20,
        max_value=0.20,
        value=0.00,
        step=0.01,
        help=(
            "Adds the same offset to every saved functional-group threshold. "
            "This is for inspection only; calibrate thresholds per group "
            "using validation data before using adjusted thresholds as final."
        )
    )
    return adjustment


# =========================================================
# 10. LOAD PROJECT DATA AND MODEL
# =========================================================

try:
    X, Y, metadata = load_dataset()
    saved_model = load_model()
    labels = list(saved_model["labels"])

    # Verify saved selected features against X_features.csv.
    selected_features = list(saved_model["selected_features"])

    if not all(
        isinstance(item, (int, np.integer))
        for item in selected_features
    ):
        missing_features = [
            feature for feature in selected_features
            if feature not in X.columns
        ]
        if missing_features:
            raise ValueError(
                f"{len(missing_features)} selected model features "
                "are missing from X_features.csv."
            )
    else:
        if min(selected_features) < 0 or max(selected_features) >= X.shape[1]:
            raise ValueError(
                "Saved feature indices do not match X_features.csv."
            )

    PROJECT_READY = True
    LOAD_ERROR = None

except Exception as error:
    PROJECT_READY = False
    LOAD_ERROR = str(error)
    X = Y = metadata = None
    labels = []


# =========================================================
# 11. SIDEBAR
# =========================================================

st.sidebar.title("🧪 IR Spectroscopy Analyzer")

page = st.sidebar.radio(
    "Select a page",
    [
        "Project Overview",
        "Known Molecule Analysis",
        "Upload IR Spectrum"
    ]
)

threshold_adjustment = show_threshold_control()


# =========================================================
# 12. MAIN HEADER AND ERROR HANDLING
# =========================================================

st.title("AI-Based IR Spectroscopy Analyzer")

if not PROJECT_READY:
    st.error(f"Project loading error: {LOAD_ERROR}")

    with st.expander("Project file locations"):
        st.write("Project folder:", str(BASE_DIR))
        st.write("Data folder:", str(DATA_DIR))
        st.write("X features:", str(X_PATH))
        st.write("Y labels:", str(Y_PATH))
        st.write("Metadata:", str(METADATA_PATH))
        st.write("Extra Trees model:", str(MODEL_PATH))


# =========================================================
# 13. PROJECT OVERVIEW
# =========================================================

if page == "Project Overview":
    st.header("Project Overview")

    st.write(
        "An AI-based system that uses IR spectroscopy to identify "
        "molecular functional groups."
    )

    if PROJECT_READY:
        st.success("Dataset and Extra Trees model loaded successfully.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Molecules", f"{len(X):,}")
        col2.metric("Original IR Features", X.shape[1])
        col3.metric("Functional Groups", len(labels))

        st.subheader("Model Information")
        st.write("Algorithm: Extra Trees")
        st.write("Selected features: 1000")
        st.write("Original spectral features: 3501")
        st.write("Model file:", str(MODEL_PATH))

        st.subheader("Supported Functional Groups")
        st.write(", ".join(map(str, labels)))

        st.subheader("Dataset Dimensions")
        st.write("X features:", X.shape)
        st.write("Y labels:", Y.shape)
        st.write("Molecule metadata:", metadata.shape)

        st.info(
            "The application uses the existing trained model. "
            "It does not retrain the model."
        )

        if not RDKIT_AVAILABLE:
            st.warning(
                "RDKit is unavailable. Install it with "
                "`python -m pip install rdkit` to enable structure checks."
            )


# =========================================================
# 14. KNOWN MOLECULE ANALYSIS
# =========================================================

elif page == "Known Molecule Analysis":
    st.header("Known Molecule Analysis")

    if PROJECT_READY:
        if "mol_id" not in metadata.columns:
            st.error("mol_id column is missing from molecule_metadata.csv.")

        elif "smiles" not in metadata.columns:
            st.error("smiles column is missing from molecule_metadata.csv.")

        else:
            molecule_ids = metadata["mol_id"].tolist()

            default_index = 0
            for i, molecule_id in enumerate(molecule_ids):
                try:
                    if int(molecule_id) == 1056:
                        default_index = i
                        break
                except (TypeError, ValueError):
                    continue

            selected_id = st.selectbox(
                "Select Molecule ID",
                molecule_ids,
                index=default_index
            )

            selected_positions = np.flatnonzero(
                metadata["mol_id"].astype(str).to_numpy()
                == str(selected_id)
            )

            if len(selected_positions) == 0:
                st.error("Could not find the selected molecule.")
            else:
                row_index = int(selected_positions[0])
                molecule = metadata.iloc[row_index]
                smiles = str(molecule["smiles"])

                st.subheader("Molecule Information")
                st.write("Molecule ID:", selected_id)
                st.write("SMILES:", smiles)

                if "canonical_smiles" in metadata.columns:
                    st.write(
                        "Canonical SMILES:",
                        molecule["canonical_smiles"]
                    )

                # X, Y and metadata are aligned by row after reset_index.
                spectrum = X.iloc[row_index].to_numpy(dtype=float)

                results = predict_functional_groups(
                    spectrum,
                    smiles=smiles,
                    threshold_adjustment=threshold_adjustment
                )

                detected = show_prediction_results(results)

                # Retrieve reference labels for this known dataset molecule.
                actual_labels = []
                for label in labels:
                    if label in Y.columns:
                        value = pd.to_numeric(
                            pd.Series([Y.iloc[row_index][label]]),
                            errors="coerce"
                        ).iloc[0]

                        if pd.notna(value) and int(value) == 1:
                            actual_labels.append(str(label))

                st.subheader("Actual Dataset Labels")
                if actual_labels:
                    st.write(", ".join(actual_labels))
                else:
                    st.write("No active labels recorded in Y_labels.csv.")

                show_actual_label_comparison(results, actual_labels)

                st.subheader("Molecular Structure")
                draw_molecule(smiles, actual_labels)

                if RDKIT_AVAILABLE:
                    st.subheader("SMILES Structural Cross-Check")
                    st.caption(
                        "This check uses the SMARTS patterns defined in this "
                        "application. A 'No structure match' result should be "
                        "reviewed; it does not automatically invalidate the "
                        "IR model prediction."
                    )
                    st.dataframe(
                        results[
                            [
                                "Functional Group",
                                "Prediction",
                                "SMILES Structural Check"
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )


# =========================================================
# 15. UPLOAD IR SPECTRUM
# =========================================================

elif page == "Upload IR Spectrum":
    st.header("Upload IR Spectrum")

    st.write(
        "Upload a CSV file containing one IR spectrum with 3501 spectral values."
    )

    if PROJECT_READY:
        uploaded_file = st.file_uploader(
            "Upload spectrum CSV",
            type=["csv"]
        )

        spectrum_type = st.radio(
            "Spectrum format",
            [
                "Raw spectrum",
                "Already preprocessed spectrum"
            ],
            help=(
                "Choose the format that matches the input representation "
                "used when the model was trained."
            )
        )

        smiles_input = st.text_input(
            "Optional: Enter SMILES for structure cross-check and visualization"
        )

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)

                numeric_df = uploaded_df.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                numeric_df = numeric_df.dropna(axis=1, how="all")
                numeric_df = numeric_df.dropna(axis=0, how="all")

                if numeric_df.empty:
                    raise ValueError("No numeric spectrum data found.")

                if len(numeric_df) != 1:
                    raise ValueError(
                        "Please upload a CSV containing exactly one spectrum row."
                    )

                values = numeric_df.iloc[0].dropna().to_numpy(dtype=float)

                # Remove a leading saved index column only if there are 3502 values.
                if len(values) == 3502:
                    values = values[1:]

                if len(values) != 3501:
                    raise ValueError(
                        f"Expected 3501 values, found {len(values)}."
                    )

                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        "The uploaded spectrum contains invalid values."
                    )

                if spectrum_type == "Raw spectrum":
                    spectrum = preprocess_spectrum(values)
                else:
                    spectrum = values

                st.success("Spectrum loaded successfully.")

                st.subheader("Uploaded IR Spectrum")
                chart_data = pd.DataFrame({
                    "Wavenumber (cm⁻¹)": np.arange(500, 4001),
                    "Intensity": spectrum
                }).set_index("Wavenumber (cm⁻¹)")

                st.line_chart(chart_data)

                if st.button("Predict Functional Groups"):
                    results = predict_functional_groups(
                        spectrum,
                        smiles=smiles_input.strip() or None,
                        threshold_adjustment=threshold_adjustment
                    )

                    detected = show_prediction_results(results)

                    if smiles_input.strip():
                        if RDKIT_AVAILABLE:
                            st.subheader("SMILES Structural Cross-Check")
                            st.caption(
                                "The structural check is based on the supplied "
                                "SMILES and the SMARTS patterns in this app. "
                                "It is a supporting check, not a replacement "
                                "for the IR prediction."
                            )
                            st.dataframe(
                                results[
                                    [
                                        "Functional Group",
                                        "Prediction",
                                        "SMILES Structural Check"
                                    ]
                                ],
                                use_container_width=True,
                                hide_index=True
                            )

                            st.subheader("Molecular Structure")
                            draw_molecule(
                                smiles_input.strip(),
                                detected["Functional Group"].tolist()
                            )
                        else:
                            st.warning(
                                "RDKit is not installed, so the SMILES "
                                "structural check is unavailable."
                            )

            except Exception as error:
                st.error(f"Could not process spectrum: {error}")


# =========================================================
# 16. SIDEBAR FOOTER
# =========================================================

st.sidebar.divider()
st.sidebar.caption(
    "AI-Based Functional Group Identification using IR Spectroscopy"
)