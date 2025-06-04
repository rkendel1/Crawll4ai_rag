# Pydantic AI Weather Agent Example

This is the complete weather agent example from the Pydantic AI documentation. It demonstrates:

- Using tools with Pydantic AI agents
- Dependency injection
- Streaming responses
- Building a Gradio UI
- Integration with external APIs

## Files

1. **`pydantic_ai_weather_agent_full.py`** - The complete weather agent implementation with:
   - `get_lat_lng` tool for geocoding locations
   - `get_weather` tool for fetching weather data
   - Examples of different usage patterns (simple queries, multiple locations, streaming)

2. **`weather_agent_gradio_ui.py`** - A Gradio web interface for the weather agent that shows:
   - Real-time streaming of agent responses
   - Tool usage visualization
   - Interactive chat interface

3. **`weather_agent_example.py`** - A simplified version using free APIs that don't require keys

## Setup

### Install Dependencies

```bash
pip install -r requirements_weather_agent.txt
```

### API Keys (Optional)

The agent works with dummy data if no API keys are provided, but for real weather data:

1. **Weather API**: Get a free key from [Tomorrow.io](https://www.tomorrow.io/weather-api/)
2. **Geocoding API**: Get a free key from [Mapbox](https://www.mapbox.com/) or [Geocode Maps](https://geocode.maps.co/)

Set them as environment variables:
```bash
export WEATHER_API_KEY="your-weather-api-key"
export GEO_API_KEY="your-geocoding-api-key"
```

Or create a `.env` file:
```
WEATHER_API_KEY=your-weather-api-key
GEO_API_KEY=your-geocoding-api-key
```

## Running the Examples

### Command Line Version
```bash
python pydantic_ai_weather_agent_full.py
```

### Gradio UI Version
```bash
python weather_agent_gradio_ui.py
```

Then open http://localhost:7860 in your browser.

### Simple Version (No API Keys Required)
```bash
python weather_agent_example.py
```

## Example Queries

- "What's the weather like in London?"
- "Compare the weather in Paris, Tokyo, and New York"
- "Is it raining in Seattle?"
- "Tell me about the weather conditions in Sydney and Melbourne"

## How It Works

1. The agent receives a natural language query about weather
2. It uses the `get_lat_lng` tool to convert location names to coordinates
3. It uses the `get_weather` tool to fetch weather data for those coordinates
4. It formats and returns a concise response

The agent automatically handles:
- Multiple locations in a single query
- Retries on failures
- Graceful fallback to dummy data when API keys are missing
