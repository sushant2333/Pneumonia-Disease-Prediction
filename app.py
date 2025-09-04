from fastai.vision.all import *
import streamlit as st
from pathlib import Path
import altair as alt


@st.cache_resource
def load_model() -> Learner:
    export_path = Path("export.pkl")
    if not export_path.exists():
        st.error("export.pkl not found. Please run training to create it.")
        st.stop()
    return load_learner(export_path)


def predict_image(learner: Learner, img_path: Path | None = None, uploaded: bytes | None = None):
    if img_path is None and uploaded is None:
        return None
    if uploaded is not None:
        img = PILImage.create(uploaded)
    else:
        img = PILImage.create(img_path)
    pred_class, pred_idx, pred_probs = learner.predict(img)
    return pred_class, pred_probs


def main() -> None:
    st.set_page_config(page_title="Pneumonia Classifier", page_icon="🫁", layout="wide")

    st.markdown(
        """
        <style>
        .app-header {font-size: 28px; font-weight: 700; margin-bottom: 0.25rem}
        .app-subtle {color: #6b7280; margin-bottom: 1.25rem}
        .pred-pill {display:inline-block;padding:6px 10px;border-radius:9999px;background:#111827;color:#fff;font-weight:600}
        .prob-chip {display:inline-block;margin:4px 6px;padding:6px 10px;border-radius:9999px;background:#eef2ff;color:#3730a3;font-weight:600}
        .footer {color:#6b7280; font-size:12px; margin-top:24px}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="app-header">🫁 Pneumonia Chest X-ray Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtle">Upload an image or choose a sample. The model predicts NORMAL vs PNEUMONIA.</div>', unsafe_allow_html=True)

    learner = load_model()
    labels = list(map(str, learner.dls.vocab))

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"]) 

        test_dir = Path(r"D:\Projects\pneumonia disease prediction\chest_xray\test")
        sample_files = get_image_files(test_dir)
        sample_choice = None
        if len(sample_files) > 0:
            sample_choice = st.selectbox("Or pick a sample", options=["(none)"] + [str(p) for p in sample_files[:100]])

        st.divider()
        tta = st.toggle("Enable TTA (Test-Time Augmentation)", value=False, help="Average predictions over augmented views")
        threshold = st.slider("Decision threshold for PNEUMONIA", 0.1, 0.9, 0.5, 0.05)

    # Main columns: left image preview, right controls + results
    left, right = st.columns([5, 7])

    # Resolve input image
    selected_path: Path | None = None
    if sample_choice and sample_choice != "(none)":
        selected_path = Path(sample_choice)

    # Preview image
    with left:
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded X-ray", use_container_width=True)
        elif selected_path is not None and selected_path.exists():
            st.image(str(selected_path), caption=selected_path.name, use_container_width=True)
        else:
            st.info("Upload an image or select a sample to preview here.")

    # Run prediction (button outside sidebar, top-right)
    with right:
        col_a, col_b = st.columns([5, 2])
        with col_b:
            run = st.button("Predict", type="primary", use_container_width=True)

    if 'run' in locals() and run:
        if uploaded_file is None and selected_path is None:
            st.warning("Please upload an image or select a sample.")
        else:
            if tta:
                # Use TTA by averaging several augmented predictions
                # Note: fastai's built-in tta is for dataloaders; here we simulate by averaging flips.
                img = PILImage.create(uploaded_file.getvalue() if uploaded_file else selected_path)
                preds = []
                for flip in [False, True]:
                    aug_img = img.flip_lr() if flip else img
                    pred_class, pred_idx, pred_probs = learner.predict(aug_img)
                    preds.append(pred_probs)
                pred_probs = torch.stack(preds).mean(dim=0)
                pred_idx = int(torch.argmax(pred_probs))
                pred_class = learner.dls.vocab[pred_idx]
            else:
                result = predict_image(learner, img_path=selected_path, uploaded=uploaded_file.getvalue() if uploaded_file else None)
                pred_class, pred_probs = result  # type: ignore
                pred_idx = int(torch.argmax(pred_probs))

            pneumonia_prob = float(pred_probs[labels.index("PNEUMONIA")]) if "PNEUMONIA" in labels else float(pred_probs[pred_idx])
            final_class = "PNEUMONIA" if pneumonia_prob >= threshold else "NORMAL"

            with right:
                st.markdown(f"<span class='pred-pill'>Prediction: {final_class}</span>", unsafe_allow_html=True)
                st.caption(f"Model top class: {pred_class} — PNEUMONIA probability: {pneumonia_prob:.3f} (threshold {threshold:.2f})")

                # Probabilities chart
                prob_data = [{"label": labels[i], "prob": float(pred_probs[i])} for i in range(len(labels))]
                chart = alt.Chart(alt.Data(values=prob_data)).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("label:N", title="Class"),
                    y=alt.Y("prob:Q", title="Probability", scale=alt.Scale(domain=[0,1])),
                    color=alt.Color("label:N", legend=None)
                ).properties(height=240)
                st.altair_chart(chart, use_container_width=True)

                # Chips
                st.markdown(" ".join([f"<span class='prob-chip'>{d['label']}: {d['prob']:.3f}</span>" for d in prob_data]), unsafe_allow_html=True)

    with st.expander("Confusion matrices"):
        cm_val_path = Path("confusion_matrix.png")
        cm_test_path = Path("confusion_matrix_test.png")
        if cm_val_path.exists():
            st.image(str(cm_val_path), caption="Validation Confusion Matrix", use_container_width=True)
        if cm_test_path.exists():
            st.image(str(cm_test_path), caption="Test Confusion Matrix", use_container_width=True)

    st.markdown("<div class='footer'>Model: ResNet50 (fastai v2). This app does not provide medical advice.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()


