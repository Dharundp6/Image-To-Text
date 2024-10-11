import gradio as gr
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
from PIL import Image

# Load BLIP model for question-answering
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

def ask_question(image, question):
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def generate_furniture_description(image):
    if image is None:
        return "No image uploaded. Please upload an image of furniture."

    # Ensure the image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Prepare questions
    questions = [
        "What type of furniture is this?",
        "What material is this furniture made of?",
        "What is the primary color of this furniture?",
        "What is the style of this furniture?",
        "What is the condition of this furniture?",
        "What are the key features of this furniture?",
        "What room would this furniture be suitable for?",
        "Are there any patterns or designs on this furniture?",
        "Does this furniture have any visible textures?",
        "What is the approximate size of this furniture?"
    ]

    # Get answers
    answers = [ask_question(image, q) for q in questions]

    # Construct the description
    description = f"""
    Furniture Description:

    Type: {answers[0]}
    Material: {answers[1]}
    Primary Color: {answers[2]}
    Style: {answers[3]}
    Condition: {answers[4]}
    Key Features: {answers[5]}
    Suitable Room: {answers[6]}
    Patterns/Designs: {answers[7]}
    Textures: {answers[8]}
    Approximate Size: {answers[9]}

    Additional Details:
    This {answers[0].lower()} appears to be made of {answers[1].lower()}. It has a predominant {answers[2].lower()} color and showcases a {answers[3].lower()} style. The furniture seems to be in {answers[4].lower()} condition.

    Notable features include {answers[5].lower()}. This piece would likely suit a {answers[6].lower()} setting. {answers[7]} {answers[8].lower() if 'no' not in answers[8].lower() else ''}

    Based on the visual analysis, the approximate size is {answers[9].lower()}.

    Note: This description is based on AI analysis and may not capture all nuances of the furniture piece.
    """

    return description

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Enhanced Furniture Description AI using BLIP")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Furniture Image")
            describe_button = gr.Button("Describe Furniture")
        with gr.Column():
            output = gr.Textbox(label="Furniture Description", lines=20)
    
    describe_button.click(fn=generate_furniture_description, inputs=image_input, outputs=output)

if __name__ == "__main__":
    demo.launch()