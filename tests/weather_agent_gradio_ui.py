from __future__ import annotations as _annotations
import json
import os
from httpx import AsyncClient
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai_weather_agent_full import Deps, weather_agent

try:
    import gradio as gr
except ImportError as e:
    raise ImportError(
        'Please install gradio with `pip install gradio`. You must use python>=3.10.'
    ) from e

TOOL_TO_DISPLAY_NAME = {'get_lat_lng': 'Geocoding API', 'get_weather': 'Weather API'}

client = AsyncClient()
weather_api_key = os.getenv('WEATHER_API_KEY')
# create a free API key at https://geocode.maps.co/
geo_api_key = os.getenv('GEO_API_KEY')
deps = Deps(client=client, weather_api_key=weather_api_key, geo_api_key=geo_api_key)


async def stream_from_agent(prompt: str, chatbot: list[dict], past_messages: list):
    chatbot.append({'role': 'user', 'content': prompt})
    yield gr.Textbox(interactive=False, value=''), chatbot, gr.skip()
    
    async with weather_agent.run_stream(
        prompt, deps=deps, message_history=past_messages
    ) as result:
        for message in result.new_messages():
            for call in message.parts:
                if isinstance(call, ToolCallPart):
                    call_args = call.args_as_json_str()
                    metadata = {
                        'title': f'🛠️ Using {TOOL_TO_DISPLAY_NAME[call.tool_name]}',
                    }
                    if call.tool_call_id is not None:
                        metadata['id'] = call.tool_call_id
                    gr_message = {
                        'role': 'assistant',
                        'content': 'Parameters: ' + call_args,
                        'metadata': metadata,
                    }
                    chatbot.append(gr_message)
                    
                if isinstance(call, ToolReturnPart):
                    for gr_message in chatbot:
                        if (
                            gr_message.get('metadata', {}).get('id')
                            == call.tool_call_id
                        ):
                            gr_message['content'] += f'\n\nResponse: {call.content}'
                            break
                    
            yield gr.skip(), chatbot, gr.skip()
            
        response_text = ''
        async for text in result.stream_text():
            response_text += text
            chatbot.append({'role': 'assistant', 'content': response_text})
            yield gr.skip(), chatbot, gr.skip()
            chatbot.pop()
            
        chatbot.append({'role': 'assistant', 'content': response_text})
        yield gr.Textbox(interactive=True, value=''), chatbot, result.all_messages()


def create_ui():
    """Create the Gradio UI for the weather agent."""
    with gr.Blocks(title='Weather Agent') as demo:
        gr.Markdown(
            """
            # 🌤️ Weather Agent
            
            Ask me about the weather in any location! I can tell you about:
            - Current temperature
            - Weather conditions
            - Humidity and wind speed
            
            Try asking: "What's the weather like in Paris?" or "Compare the weather in London and Tokyo"
            """
        )
        
        chatbot = gr.Chatbot(
            label='Weather Assistant',
            type='messages',
            height=500,
            show_copy_button=True,
        )
        
        with gr.Row():
            prompt = gr.Textbox(
                label='Ask about weather',
                placeholder='What is the weather like in New York?',
                lines=1,
                scale=4,
            )
            submit_btn = gr.Button('Send', variant='primary', scale=1)
        
        past_messages = gr.State([])
        
        # Handle submit
        submit_btn.click(
            stream_from_agent,
            inputs=[prompt, chatbot, past_messages],
            outputs=[prompt, chatbot, past_messages],
        )
        
        # Handle enter key
        prompt.submit(
            stream_from_agent,
            inputs=[prompt, chatbot, past_messages],
            outputs=[prompt, chatbot, past_messages],
        )
        
        # Add examples
        gr.Examples(
            examples=[
                "What's the weather like in London?",
                "Compare the weather in Paris, Tokyo, and New York",
                "Is it raining in Seattle?",
                "What's the temperature in Dubai?",
                "Tell me about the weather conditions in Sydney and Melbourne",
            ],
            inputs=prompt,
        )
    
    return demo


if __name__ == '__main__':
    demo = create_ui()
    demo.launch(share=False, server_name='0.0.0.0', server_port=7860)
