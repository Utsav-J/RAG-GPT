import gradio as gr
from utils.chatbot import Chatbot
from utils.ui_settings import UISettings
from utils.summarizer import Summarizer 
from utils.upload_file import UploadFile

with gr.Block() as demo:
    with gr.Tabs():
        with gr.TabItem('RAG-GPT'):
            ##############
            # First ROW:
            ##############
            with gr.Row() as row1:
                with gr.Column(visible=False) as reference_bar:
                    ref_output = gr.Markdown()

                with gr.Column() as chatbot_output:
                    chatbot = gr.Chatbot(
                        [],
                        elem_id = 'Chatbot',
                        bubble_full_width=False,
                        height=500,
                        avatar_images=None # TODO - add a new image
                    )
                    Chatbot.like(UISettings.feedback, None, None)
            ##############
            # SECOND ROW:
            ##############
            with gr.Row():
                input_text = gr.Textbox(
                    lines=4,
                    scale=8,
                    placeholder="Enter text and press enter or upload the PDF files",
                    container=False
                )
            ##############
            # Third ROW:
            ##############

            with gr.Row() as row_two:
                text_submit_btn = gr.Button(value="Submit text")
                sidebar_state = gr.State(False)
                btn_toggle_sidebar = gr.Button(value="References")
                btn_toggle_sidebar.click(UISettings.toggle_sidebar, [sidebar_state], [reference_bar, sidebar_state])
                upload_btn = gr.UploadButton(
                        "📁 Upload PDF or doc files", file_types=[
                        '.pdf',
                        '.doc'
                    ],
                    file_count="multiple")
                temperature_bar = gr.Slider(minimum=0, maximum=1, value=0, step=0.1,label="Temperature", info="Choose between 0 and 1")
                rag_with_dropdown = gr.Dropdown(label="RAG with", choices=["Preprocessed doc", "Upload doc: Process for RAG", "Upload doc: Give Full summary"], value="Preprocessed doc")
                clear_button = gr.ClearButton([input_text, Chatbot])
            ##############
            # Process:
            ##############
            file_msg = upload_btn.upload(fn = UploadFile.process_uploaded_files, 
                                         inputs=[upload_btn,chatbot,rag_with_dropdown],
                                         outputs=[input_text,chatbot],
                                         queue=False)
            txt_msg = input_text.submit(fn=Chatbot.respond,
                                       inputs=[Chatbot, input_text,
                                               rag_with_dropdown, temperature_bar],
                                       outputs=[input_text,
                                                Chatbot, ref_output],
                                       queue=False).then(lambda: gr.Textbox(interactive=True),
                                                         None, [input_text], queue=False)

            txt_msg = text_submit_btn.click(fn=Chatbot.respond,
                                            inputs=[Chatbot, input_text,
                                                    rag_with_dropdown, temperature_bar],
                                            outputs=[input_text,
                                                     Chatbot, ref_output],
                                            queue=False).then(lambda: gr.Textbox(interactive=True),
                                                              None, [input_text], queue=False)


if __name__ == "__main__":
    demo.launch()