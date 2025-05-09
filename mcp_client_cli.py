import typer
from typing import Optional, List, Dict, Any
import json
import subprocess
import asyncio
import os
import sys
from pathlib import Path
import shlex # For safely parsing command strings if needed

# Assuming the mcp-sdk is installed in your environment
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
# Import correct MCP types - these may need adjustment based on your installed version
# from mcp.types import ToolCallRequest, ToolCallResponse, ToolInput, ServerInfo

# --- Configuration ---
CONFIG_FILE_NAME = "cli_config.json"
app = typer.Typer(help="CLI to test and interact with MCP servers.")

# --- Helper Functions ---
def load_cli_config() -> Dict[str, Any]:
    """Loads the CLI configuration for MCP servers."""
    config_path = Path(__file__).parent / CONFIG_FILE_NAME
    if not config_path.exists():
        typer.secho(f"Error: Configuration file '{CONFIG_FILE_NAME}' not found in {config_path.parent}.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        typer.secho(f"Error: Could not parse '{CONFIG_FILE_NAME}'. Invalid JSON.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

def get_server_launch_config(server_name: str) -> Dict[str, Any]:
    """Gets the launch configuration for a specific server."""
    config_data = load_cli_config()
    servers = config_data.get("mcpServers", {})
    if server_name not in servers:
        typer.secho(f"Error: Server '{server_name}' not found in '{CONFIG_FILE_NAME}'.", fg=typer.colors.RED)
        typer.secho(f"Available servers: {list(servers.keys())}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    return servers[server_name]

async def read_stderr(stderr_pipe, server_name: str):
    """Reads from server's stderr and prints it."""
    while True:
        line = await stderr_pipe.readline()
        if not line:
            break
        typer.secho(f"[{server_name} STDERR] {line.decode().strip()}", fg=typer.colors.MAGENTA)

# --- CLI Commands ---

@app.command()
def list_defined_servers():
    """Lists all servers defined in the CLI configuration."""
    config_data = load_cli_config()
    servers = config_data.get("mcpServers", {})
    if not servers:
        typer.secho("No servers defined in the configuration.", fg=typer.colors.YELLOW)
        return

    typer.secho("Defined MCP Servers:", bold=True)
    for name, details in servers.items():
        typer.echo(f"- {typer.style(name, fg=typer.colors.GREEN)}")
        typer.echo(f"  Command: {details.get('command')}")
        typer.echo(f"  Args: {details.get('args')}")
        typer.echo(f"  CWD: {details.get('cwd', 'Not set (uses CLI CWD or script location)')}")

async def _manage_server_and_client(server_name: str, config: Dict[str, Any]):
    """Manages server process and MCP client connection."""
    command = config["command"]
    args = config.get("args", [])
    cwd = config.get("cwd") # Can be None

    if cwd:
        cwd = Path(cwd).resolve() # Ensure absolute path
        if not cwd.is_dir():
            typer.secho(f"Error: Specified CWD for server '{server_name}' does not exist or is not a directory: {cwd}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
            
    typer.secho(f"Attempting to start server '{server_name}'...", fg=typer.colors.CYAN)
    typer.secho(f"  Command: {command} {' '.join(shlex.quote(str(c)) for c in args)}", fg=typer.colors.CYAN)
    if cwd:
         typer.secho(f"  CWD: {cwd}", fg=typer.colors.CYAN)

    process = None
    client = None
    stderr_task = None

    try:
        # Create server parameters
        server_params = StdioServerParameters(
            command=command, 
            args=args,
            cwd=cwd
        )
        
        # Connect to the server using stdio_client
        async with stdio_client(server_params) as (read_stream, write_stream):
            typer.secho(f"Server '{server_name}' started.", fg=typer.colors.GREEN)
            
            # Create client session
            client = ClientSession(read_stream, write_stream)
            await client.initialize()
            
            typer.secho("MCP Connection Initialized Successfully!", fg=typer.colors.GREEN)
            typer.secho(f"  Server Name: {server_name}", fg=typer.colors.BLUE)
            
            yield client # Yield the client to the command function

    except FileNotFoundError:
        typer.secho(f"Error: Command '{command}' not found. Is it in your PATH or an absolute path?", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except ConnectionRefusedError: # Or other MCP connection errors
        typer.secho(f"Error: Could not connect to MCP server '{server_name}'. Is it running correctly?", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

@app.command()
def list_tools(server_name: str = typer.Argument(..., help="Name of the server as defined in cli_config.json.")):
    """Lists available tools from the specified MCP server."""
    # Run the async function in an event loop
    asyncio.run(_list_tools_async(server_name))

async def _list_tools_async(server_name: str):
    """Async implementation of list_tools command."""
    config = get_server_launch_config(server_name)
    async for client in _manage_server_and_client(server_name, config):
        tools = await client.list_tools()
        if not tools:
            typer.secho("No tools reported by the server.", fg=typer.colors.YELLOW)
            return

        typer.secho(f"\nAvailable tools from '{server_name}':", bold=True)
        for tool_name, tool_def in tools.items():
            typer.secho(f"- {typer.style(tool_name, fg=typer.colors.GREEN)}", bold=True)
            typer.echo(f"  Description: {tool_def.description}")
            if hasattr(tool_def, 'input_schema') and tool_def.input_schema and hasattr(tool_def.input_schema, 'properties'):
                typer.echo("  Input Schema:")
                for prop_name, prop_def in tool_def.input_schema.properties.items():
                    typer.echo(f"    {prop_name} ({prop_def.get('type', 'any')}): {prop_def.get('description', 'No description')}")
            else:
                typer.echo("  Input Schema: None")

@app.command()
def call_tool(
    server_name: str = typer.Argument(..., help="Name of the server."),
    tool_name: str = typer.Argument(..., help="Name of the tool to call."),
    tool_args_json: str = typer.Option("{}", help="JSON string of arguments for the tool. E.g., '{\"param1\": \"value1\"}'")
):
    """Calls a specific tool on the MCP server with given arguments."""
    # Run the async function in an event loop
    asyncio.run(_call_tool_async(server_name, tool_name, tool_args_json))

async def _call_tool_async(server_name: str, tool_name: str, tool_args_json: str):
    """Async implementation of call_tool command."""
    config = get_server_launch_config(server_name)
    
    try:
        parsed_args = json.loads(tool_args_json)
        if not isinstance(parsed_args, dict):
            typer.secho("Error: Tool arguments must be a JSON object (dictionary).", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    except json.JSONDecodeError:
        typer.secho(f"Error: Invalid JSON provided for tool arguments: {tool_args_json}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    async for client in _manage_server_and_client(server_name, config):
        typer.secho(f"\nCalling tool '{tool_name}' on server '{server_name}' with args: {parsed_args}", fg=typer.colors.CYAN)
        try:
            response = await client.call_tool(tool_name, parsed_args)
            
            typer.secho("Tool Call Successful!", fg=typer.colors.GREEN)
            typer.secho("Response Content:", bold=True)
            if response and hasattr(response, 'content'):
                for content_item in response.content:
                    if hasattr(content_item, 'type') and content_item.type == "text" and hasattr(content_item, 'text') and content_item.text is not None:
                        # Attempt to parse if it's JSON, otherwise print as text
                        try:
                            json_output = json.loads(content_item.text)
                            typer.echo(json.dumps(json_output, indent=2))
                        except json.JSONDecodeError:
                             typer.echo(content_item.text)
                    else:
                        typer.echo(f"  Type: {getattr(content_item, 'type', 'unknown')}, Data: {str(content_item)[:200]}...") # Truncate potentially large data
            else:
                typer.secho("  (No content returned by tool)", fg=typer.colors.YELLOW)

        except Exception as e: # Catch specific MCP errors if known, e.g., ToolNotFound
            typer.secho(f"Error calling tool '{tool_name}': {e}", fg=typer.colors.RED)

if __name__ == "__main__":
    # Create config file if it doesn't exist, with a placeholder
    config_path_check = Path(__file__).parent / CONFIG_FILE_NAME
    if not config_path_check.exists():
        placeholder_config = {
            "mcpServers": {
                "my_server_example": {
                    "command": "python",
                    "args": ["/path/to/your/mcp_server.py"],
                    "cwd": "/path/to/your/project_root"
                }
            }
        }
        with open(config_path_check, 'w') as f:
            json.dump(placeholder_config, f, indent=4)
        print(f"Created placeholder config: {config_path_check}. Please edit it.")

    # Run the Typer app normally (will use the sync functions that wrap async)
    app()