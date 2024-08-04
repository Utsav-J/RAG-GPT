import gradio as gr

def say_hi(name):
    return "Hello " + name + " !"

with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output")
    greet_button  =gr.Button("Greet")
    greet_button.click(fn = say_hi, inputs = name, outputs=output, api_name="greet")


demo.launch()
