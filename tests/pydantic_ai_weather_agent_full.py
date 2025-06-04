from __future__ import annotations as _annotations
import asyncio
import os
import urllib.parse
from dataclasses import dataclass
from typing import Any
import logfire
from devtools import debug
from httpx import AsyncClient
from pydantic_ai import Agent, ModelRetry, RunContext

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


@dataclass
class Deps:
    client: AsyncClient
    weather_api_key: str | None
    geo_api_key: str | None


weather_agent = Agent(
    'openai:gpt-4o',
    # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
    # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
    instructions=(
        'Be concise, reply with one sentence.'
        'Use the `get_lat_lng` tool to get the latitude and longitude of the locations, '
        'then use the `get_weather` tool to get the weather.'
    ),
    deps_type=Deps,
    retries=2,
)


@weather_agent.tool
async def get_lat_lng(
    ctx: RunContext[Deps], location_description: str
) -> dict[str, float]:
    """Get the latitude and longitude of a location.
    Args:
        ctx: The context.
        location_description: A description of a location.
    """
    if ctx.deps.geo_api_key is None:
        # if no API key is provided, return a dummy response (London)
        return {'lat': 51.1, 'lng': -0.1}
    
    params = {'access_token': ctx.deps.geo_api_key}
    loc = urllib.parse.quote(location_description)
    r = await ctx.deps.client.get(
        f'https://api.mapbox.com/geocoding/v5/mapbox.places/{loc}.json', params=params
    )
    r.raise_for_status()
    data = r.json()
    
    # Extract the first feature if available
    if data.get('features'):
        feature = data['features'][0]
        lng, lat = feature['geometry']['coordinates']
        return {'lat': lat, 'lng': lng}
    else:
        raise ModelRetry(f'Could not find location: {location_description}')


@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location.
    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    if ctx.deps.weather_api_key is None:
        # if no API key is provided, return a dummy response
        return {'temperature': '21°C', 'description': 'Sunny'}
    
    params = {
        'location': f'{lat},{lng}',
        'apikey': ctx.deps.weather_api_key,
        'units': 'metric',
    }
    r = await ctx.deps.client.get(
        'https://api.tomorrow.io/v4/weather/realtime', params=params
    )
    r.raise_for_status()
    data = r.json()
    
    # Extract weather data
    values = data['data']['values']
    return {
        'temperature': f"{values['temperature']}°C",
        'description': _weather_code_to_description(values.get('weatherCode', 0)),
        'humidity': f"{values.get('humidity', 0)}%",
        'wind_speed': f"{values.get('windSpeed', 0)} m/s",
    }


def _weather_code_to_description(code: int) -> str:
    """Convert weather code to human-readable description."""
    weather_codes = {
        0: 'Unknown',
        1000: 'Clear',
        1001: 'Cloudy',
        1100: 'Mostly Clear',
        1101: 'Partly Cloudy',
        1102: 'Mostly Cloudy',
        2000: 'Fog',
        2100: 'Light Fog',
        3000: 'Light Wind',
        3001: 'Wind',
        3002: 'Strong Wind',
        4000: 'Drizzle',
        4001: 'Rain',
        4200: 'Light Rain',
        4201: 'Heavy Rain',
        5000: 'Snow',
        5001: 'Flurries',
        5100: 'Light Snow',
        5101: 'Heavy Snow',
        6000: 'Freezing Drizzle',
        6001: 'Freezing Rain',
        6200: 'Light Freezing Rain',
        6201: 'Heavy Freezing Rain',
        7000: 'Ice Pellets',
        7101: 'Heavy Ice Pellets',
        7102: 'Light Ice Pellets',
        8000: 'Thunderstorm',
    }
    return weather_codes.get(code, 'Unknown')


async def main():
    """Run the weather agent with example queries."""
    async with AsyncClient() as client:
        # Create a free API key at https://www.tomorrow.io/weather-api/
        weather_api_key = os.getenv('WEATHER_API_KEY')
        # Create a free API key at https://geocode.maps.co/
        geo_api_key = os.getenv('GEO_API_KEY')
        
        deps = Deps(
            client=client,
            weather_api_key=weather_api_key,
            geo_api_key=geo_api_key
        )
        
        # Example 1: Simple weather query
        result = await weather_agent.run(
            'What is the weather like in London?',
            deps=deps,
        )
        print(result.data)
        debug(result)
        
        # Example 2: Multiple locations
        result = await weather_agent.run(
            'What is the weather like in London, New York and Tokyo?',
            deps=deps,
        )
        print(result.data)
        
        # Example 3: Streaming response
        async with weather_agent.run_stream(
            'Tell me about the weather in Paris and Madrid',
            deps=deps,
        ) as result:
            async for text in result.stream_text():
                print(text, end='', flush=True)
            print()


if __name__ == '__main__':
    asyncio.run(main())
