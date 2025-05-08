FROM python:alpine

ARG PORT=8051

WORKDIR /app

# Install uv
RUN pip install uv

# Copy the MCP server files
COPY . .

# Install packages directly to the system (no virtual environment)
# Combining commands to reduce Docker layers
RUN uv pip install --system -e . && \
    uv run python -m playwright install --with-deps chromium && \
    crawl4ai-setup

EXPOSE ${PORT}

# Command to run the MCP server
CMD ["uv", "run", "app.py"]