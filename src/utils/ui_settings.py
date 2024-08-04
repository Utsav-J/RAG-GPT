import gradio as gr

class UISettings:
    @staticmethod
    def toggle_state(state):
        state = not state
        return gr.update(visible=state), state
    
    @staticmethod
    def feedback(data: gr.LikeData):
        if data.liked:
            print("You upvoted this result" + data.value)
        else:
            print("You downvoted this result" + data.value)