import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the trained model from Hugging Face
model_name = "./" 
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the summarization function
def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Create Gradio UI
iface = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=5, placeholder="Enter text to summarize..."),
    outputs=gr.Textbox(label="Summarized Text"),
    title="Text Summarization with BART",
    description="Enter an article and get a summarized version instantly.",
)

# Launch the app
iface.launch()
