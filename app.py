# ============================================
# PERSONALIZED FITNESS ACTIVITY RECOGNITION
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Fitness Activity Recognition",
    page_icon="🏃",
    layout="wide"
)


# ============================================
# LOAD TRAINED MODEL
# ============================================

MODEL_PATH = "models/activity_model.pkl"
SCALER_PATH = "models/scaler.pkl"
MAPPING_PATH = "models/activity_mapping.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
activity_mapping = joblib.load(MAPPING_PATH)


# ============================================
# TITLE
# ============================================

st.title("🏃 Personalized Fitness Activity Recognition")

st.write(
    "Machine Learning based human activity recognition "
    "using smartphone sensor data."
)


# ============================================
# ABOUT PROJECT
# ============================================

st.divider()

st.subheader("📱 About the Project")

st.write(
    """
    This application uses smartphone sensor data such as
    accelerometer and gyroscope features to recognize
    different human activities using a trained machine
    learning model.
    """
)


# ============================================
# SUPPORTED ACTIVITIES
# ============================================

st.subheader("🎯 Supported Activities")

activities = list(activity_mapping.values())

for activity in activities:
    st.write("•", activity)


# ============================================
# PREDICTION
# ============================================

st.divider()

st.subheader("🔮 Activity Prediction")

st.info(
    "Upload a CSV containing the same 561 sensor features "
    "used during model training."
)


# ============================================
# CSV UPLOAD
# ============================================

uploaded_file = st.file_uploader(
    "Upload sensor feature CSV",
    type=["csv"]
)


# ============================================
# PROCESS UPLOADED FILE
# ============================================

if uploaded_file is not None:

    # Read CSV
    data = pd.read_csv(uploaded_file)

    st.success("CSV uploaded successfully!")

    # Show shape
    st.write("Uploaded data shape:")
    st.write(data.shape)

    # ========================================
    # SHOW UPLOADED DATA
    # ========================================

    st.subheader("📄 Uploaded Data")

    st.dataframe(
        data.head(),
        use_container_width=True
    )

    # ========================================
    # VALIDATE FEATURES
    # ========================================

    if data.shape[1] != 561:

        st.error(
            f"Invalid CSV. The model expects 561 features, "
            f"but your file contains {data.shape[1]} columns."
        )

    else:

        st.success("✅ CSV format is valid!")

        # ====================================
        # SCALE INPUT DATA
        # ====================================

        scaled_data = scaler.transform(data)

        st.success("✅ Sensor data scaled successfully!")

        # ====================================
        # MAKE PREDICTIONS
        # ====================================

        predictions = model.predict(scaled_data)

        # ====================================
        # CONVERT PREDICTIONS
        # ====================================

        predicted_activities = []

        for prediction in predictions:

            # Model already returns activity name
            if isinstance(prediction, str):

                predicted_activities.append(prediction)

            # Model returns numeric activity ID
            else:

                predicted_activities.append(
                    activity_mapping.get(
                        int(prediction),
                        str(prediction)
                    )
                )

        # ====================================
        # PREDICTION CONFIDENCE
        # ====================================

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                scaled_data
            )

            confidence_scores = (
                probabilities.max(axis=1) * 100
            )

        else:

            confidence_scores = [
                None
            ] * len(predictions)

        # ====================================
        # RESULT DATAFRAME
        # ====================================

        result_df = pd.DataFrame({
            "Predicted Activity": predicted_activities,
            "Confidence (%)": [
                round(score, 2)
                if score is not None
                else "N/A"
                for score in confidence_scores
            ]
        })

        # ====================================
        # DISPLAY RESULTS
        # ====================================

        st.subheader("🔮 Predicted Activities")

        st.dataframe(
            result_df,
            use_container_width=True
        )

        # ====================================
        # ACTIVITY SUMMARY
        # ====================================

        st.subheader("📊 Activity Summary")

        activity_counts = pd.Series(
            predicted_activities
        ).value_counts()

        st.bar_chart(activity_counts)

        # ====================================
        # PREDICTION STATISTICS
        # ====================================

        st.subheader("📈 Prediction Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Samples",
                len(predicted_activities)
            )

        with col2:

            st.metric(
                "Different Activities",
                len(set(predicted_activities))
            )

        with col3:

            if confidence_scores[0] is not None:

                avg_confidence = (
                    sum(confidence_scores)
                    / len(confidence_scores)
                )

                st.metric(
                    "Average Confidence",
                    f"{avg_confidence:.2f}%"
                )

        # ====================================
        # DOWNLOAD RESULTS
        # ====================================

        st.subheader("📥 Download Results")

        csv_result = result_df.to_csv(
            index=False
        )

        st.download_button(
            label="📥 Download Prediction Results",
            data=csv_result,
            file_name="fitness_activity_predictions.csv",
            mime="text/csv"
        )