# pcm
PCM (MCP but reversed), **MCP for reverse engineering**.

## Features

- Analysis
  - IDA
- Memory
  - Engagement reports

## Installations

Prerequisites:
- [`uv`](https://github.com/astral-sh/uv)


1. Clone the repository
    ```
    git clone https://github.com/rand-tech/pcm
    ```
1. Add `pcm` to you mcp config
    example
    ```
    {
        "mcpServers": {
            "pcm": {
                "command": "uv",
                "args": [
                    "--directory",
                    "path_to/pcm",
                    "run",
                    "server.py"
                ]
            }
        }
    }
    ```
1. Use the MCP 


**Related projects**:

- <https://github.com/mrexodia/ida-pro-mcp> 
- <https://github.com/MxIris-Reverse-Engineering/ida-mcp-server>