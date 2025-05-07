# import os
# import asyncio
# import uvicorn
# from fastapi import FastAPI
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# from pathlib import Path
# from contextlib import asynccontextmanager
# from mcp.server.fastmcp import FastMCP, Context

# # --- Load environment variables ---
# # necessary befor loading mcp
# assert load_dotenv('.env')

# from src.server.server import mcp
# # Import all tools to register them with the MCP server
# from src.server.tools import crawl_single_page, smart_crawl_url, get_available_sources, perform_rag_query



# # --- Define lifespan context manager ---
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Start MCP server when FastAPI starts
#     print("Starting MCP server in lifespan context...")
#     mcp_task = asyncio.create_task(run_mcp_server())
    
#     # Give the MCP server a moment to start and bind to its port
#     await asyncio.sleep(1)
    
#     yield
    
#     # Cancel MCP server task when FastAPI shuts down
#     print("Shutting down MCP server...")
#     mcp_task.cancel()
#     try:
#         await mcp_task
#     except asyncio.CancelledError:
#         pass
#     print("MCP server shutdown complete")


# # --- FastMCP Server Setup ---
# app = FastAPI(lifespan=lifespan)

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# # Mount the static directory for the frontend
# app.mount("/static", StaticFiles(directory="src/static"), name="static")


# @app.get("/")
# def read_index():
#     return FileResponse("src/static/index.html")

# # Add specific route for /index
# @app.get("/index")
# def read_index_alt():
#     return FileResponse("src/static/index.html")

# # Add this with your other routes
# @app.get("/tools")
# def read_tools():
#     return FileResponse("src/static/tools.html")


# # --- Entrypoint for running MCP server via SSE ---
# async def run_mcp_server():
#     transport = os.environ["TRANSPORT"]
    
#     try:
#         if transport == "sse":
#             await mcp.run_sse_async(
#                 # host=host, port=port
#                 )
#         else:
#             await mcp.run_stdio_async()
#     except Exception as e:
#         print(f"Error starting MCP server: {e}")
#         import traceback
#         print(traceback.format_exc())
#         raise


# if __name__ == "__main__":
#     # Use a standard web port for the FastAPI application
#     web_host = "0.0.0.0"  # Typically bind to all interfaces for web servers
#     web_port = 8000  # Standard port for web applications
    
#     # Output access URL for clarity
#     print(f"\n🚀 FastAPI application running at http://localhost:{web_port}")
#     print(f"🔌 MCP server running at http://localhost:{os.getenv('PORT', 8051)}/sse\n")
    
#     # Run with uvicorn directly
#     uvicorn.run(app, host=web_host, port=web_port)
