# perplexity

## how to use other mcp servers in my python streamlit code

To use other MCP (Model Context Protocol) servers in your Python Streamlit code, you need to set up your Streamlit app as an MCP client that can connect to one or more MCP servers. Here's a step-by-step guide based on the latest community practices and open-source projects:

### 1. Understand the Architecture

*   Streamlit serves as your front-end UI.
*   MCP servers handle the backend logic (LLMs, tools, etc.).
*   Communication is typically via HTTP (REST or SSE) or WebSocket APIs [1] [2] [3].

### 2. Configure Multiple MCP Servers

*   Use a configuration file (commonly `config.json` or similar) to list the URLs and credentials for each MCP server you want to use[4] [2] .
*   Example `config.json` structure:

    json
    {
      "servers": [
        {
          "name": "Local MCP",
          "url": "http://localhost:8000/mcp"
        },
        {
          "name": "Remote MCP",
          "url": "https://mcp.example.com/mcp"
        }
      ]
    }
    

### 3. Load Server Config in Streamlit

*   Read the config file at app startup and present server options to the user:

    python
    import json
    import streamlit as st
    
    with open("config.json") as f:
        servers = json.load(f) ["servers"]
    
    server_names = [s["name"] for s in servers]
    selected = st.selectbox("Select MCP Server", server_names)
    server_url = next(s["url"] for s in servers if s["name"] == selected)

## 4. Build the MCP Client Logic

- Use Python's `requests` or `httpx` to send requests to the selected MCP server[1] [5] .
- Example (using requests):

    python
    import requests
    from pydantic import BaseModel

    class Message (BaseModel):
        role: str
        content: str

    class MCPContext (BaseModel):
        user: dict
        history: list
        instructions: str

    # Collect user input
    user_id = st.text_input("User ID", value="u001")
    prompt = st.text_input("Your message")
    if st.button("Send") and prompt:
        # Build context and send to server
        context = MCPContext(
            user={"id": user_id},
            history=[Message(role="user", content=prompt)],
            instructions="Be concise."
        )
        response = requests.post(server_url, json=context.dict())
        if response.status_code == 200:
            reply = response.json().get("reply")
            st.write("Assistant:", reply)
        else:
            st.error("Failed to contact MCP server.")
    

- For advanced features (streaming, tool calls), use SSE or WebSockets as supported by your MCP server and client libraries [2] [3].

## 5. (Optional) Use Existing Open-Source Clients

- The `langchain-mcp-client` project provides a ready-made Streamlit Ul with support for multiple MCP servers, configuration management, and advanced features like memory, tool testing, and LLM provider selection [2].
- To run:
    
    streamlit run app.py
    

Then connect to your desired MCP server via the UI.

## 6. Troubleshooting and Enhancements

*   Ensure each MCP server is running and accessible from your Streamlit app [2] [5]

*   For custom tool integration, dynamically load tool definitions from each server and present them in the UI [4] [2].

*   Add authentication or API key handling if your MCP servers require it [2] [5].

## Summary Table: Key Steps

| Step                | Description                                                                                                 |
| ------------------- | ----------------------------------------------------------------------------------------------------------- |
| Configure Servers   | List MCP server URLs in a config file                                                                       |
| Load in Streamlit   | Read config, let user select server                                                                         |
| Send Requests       | Use requests/httpx to POST MCP context to chosen server                                                     |
| Display Responses   | Show assistant replies in Streamlit                                                                         |
| Advanced Features   | Use SSE/WebSockets for streaming, tool calls, etc. (as supported)                                        |
| Open-Source Option | Try `langchain-mcp-client` for a full-featured UI and multi-server support                                 |

By following these steps, you can flexibly connect your Streamlit app to any number of MCP servers and dynamically switch between them at runtime [4] [2] [5].

1.  [https://www.youtube.com/watch?v=C2Pg0nWMWew](https://www.youtube.com/watch?v=C2Pg0nWMWew)
2.  [https://github.com/guinacio/langchain-mcp-client](https://github.com/guinacio/langchain-mcp-client)
3.  [https://www.youtube.com/watch?v=0KJ2oBRtUbs](https://www.youtube.com/watch?v=0KJ2oBRtUbs)
4.  [https://www.youtube.com/watch?v=Ln-Tgz8Pmek](https://www.youtube.com/watch?v=Ln-Tgz8Pmek)
5.  [https://www.comparepriceacross.com/post/host\_mcp\_client\_or\_server\_using\_streamlit/](https://www.comparepriceacross.com/post/host_mcp_client_or_server_using_streamlit/)