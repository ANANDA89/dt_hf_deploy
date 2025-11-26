# app.py
import pickle
import numpy as np
import gradio as gr

# 1. Load models
with open("cart_model.pkl", "rb") as f:
    cart_model = pickle.load(f)

with open("id3_model.pkl", "rb") as f:
    id3_model = pickle.load(f)


# 2. Prediction function
def predict(sepal_length, sepal_width, petal_length, petal_width, model_type):
    """
    model_type: "CART (Gini)" or "ID3 (Entropy)"
    """
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if model_type == "CART (Gini)":
        clf = cart_model
    else:
        clf = id3_model

    probs = clf.predict_proba(X)[0]
    pred_class = int(np.argmax(probs))
    class_names = ["Setosa", "Versicolor", "Virginica"]

    return {
        "Predicted Class": class_names[pred_class],
        "Class 0 (Setosa)": float(probs[0]),
        "Class 1 (Versicolor)": float(probs[1]),
        "Class 2 (Virginica)": float(probs[2]),
    }


# 3. Build Gradio interface
inputs = [
    gr.Number(label="Sepal Length (cm)", value=5.1),
    gr.Number(label="Sepal Width (cm)", value=3.5),
    gr.Number(label="Petal Length (cm)", value=1.4),
    gr.Number(label="Petal Width (cm)", value=0.2),
    gr.Radio(
        choices=["CART (Gini)", "ID3 (Entropy)"],
        value="CART (Gini)",
        label="Model Type",
    ),
]

outputs = gr.JSON(label="Prediction Details")

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title="Decision Tree Classifier (CART vs ID3)",
    description="Iris classification using Decision Trees with Gini (CART) and Entropy (ID3).",
)

# ... keep all imports, model loading, predict(), inputs, outputs, demo = gr.Interface(...)

if __name__ == "__main__":
    demo.launch()
    # print("✅ Starting Gradio app...")
    # demo.launch(
    #     server_name="0.0.0.0",  # local only
    #     # server_port=7860,         # fixed port
    #     share=False               # no public link, just local
    # )
    # print("❌ Gradio app stopped")  # will print only after you CTRL+C

