import gradio as gr
import pickle
import numpy as np

# Load the KMeans model
model = pickle.load(open("model.md5", "rb"))

def predictor(ProductRelated_Duration, BounceRates):
    # Create a numpy array for prediction
    input_data = np.array([[ProductRelated_Duration, BounceRates]])

    # Use the model to predict the cluster
    prediction = model.predict(input_data)

    # Convert the prediction to human-readable labels
    if prediction[0] == 1:
        return "Target Customer"
    elif prediction[0] == 0:
        return "Uninterested Customer"

# Define the Gradio interface
iface = gr.Interface(
    fn=predictor,
    inputs=[
        gr.Slider(label="ProductRelated_Duration", minimum=0.00, maximum=65000.00, step=1),
        gr.Slider(label="BounceRates", minimum=0.00, maximum=0.2, step=0.001)
    ],
    outputs=gr.Text(),
    live=True
)

# Launch the Gradio interface
iface.launch(share=True)
