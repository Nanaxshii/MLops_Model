import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# โหลดโมเดลที่ฝึกแล้ว
model = tf.keras.models.load_model("best_model.h5")

# รายชื่อคลาสที่โมเดลสามารถจำแนกได้
class_names = ["Bacterial Dermatosis", "Fungal Infection", "Hypersensitivity Dermatitis", "Healthy"]

def skin_detection(img):
    """ทำการทำนายโรคผิวหนังของสัตว์เลี้ยงจากภาพที่อัปโหลด"""
    img = image.img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Reshape ให้เข้ากับโมเดล
    
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100  # แปลงเป็นเปอร์เซ็นต์
    
    return f"Prediction: {predicted_class} (Confidence: {confidence:.2f}%)"

examples = [('images/BacterialDermatosis.jpg', 'Bacterial Dermatosis'), ('images/FungalInfection.jpg', 'Fungal Infection'),\
             ('images/HypersensitivityDermatitis.jpg', 'Hypersensitivity Dermatitis'),('images/Healthy.jpg', 'Healthy')]

theme = gr.themes.Soft(
    secondary_hue="rose",
    neutral_hue="violet",
    radius_size="lg",
    font=[gr.themes.GoogleFont('Montserrat'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
).set(
    body_background_fill='*primary_50',
    body_background_fill_dark='*primary_950',
    body_text_color='*neutral_900',
    background_fill_primary='*neutral_100',
    background_fill_primary_dark='*primary_800',
    background_fill_secondary='*neutral_100',
    border_color_accent='*primary_950',
    border_color_accent_subdued='*primary_300',
    shadow_drop='*button_primary_shadow_active',
    shadow_drop_lg='0 5px 8px 0 rgb(0 0 0 / 0.1)',
    shadow_inset='*shadow_drop_lg',
    shadow_spread='20px',
    block_background_fill='*primary_100',
    block_background_fill_dark='*neutral_300',
    block_border_color='*border_color_accent',
    block_info_text_color_dark='*primary_800',
    block_info_text_weight='500',
    block_label_background_fill='*primary_50',
    block_label_background_fill_dark='*primary_500',
    block_shadow='*shadow_spread',
    checkbox_shadow='*shadow_drop_lg',
    button_secondary_background_fill_hover='*primary_300'
)

with gr.Blocks(theme=theme) as demo:
    ...

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# 🐶 Dog Skin Disease Detector")
            gr.Markdown("Upload an image of suspicious areas of your pet's skin for primary skin disease detection.")
            gr.Image("images/Dog paw-amico.png", width=415, height=415, show_label=False)
        with gr.Column(scale=2):
            image_input = gr.Image(type="numpy", label="Upload your pet's image")
            gr.Gallery(examples, height=80, label='Skin Disease Example')
            output_label = gr.Label(label="Your pet's skin disease Detection:")
            button = gr.Button("Detect")
            button.click(skin_detection, inputs=image_input, outputs=output_label)

demo.launch()