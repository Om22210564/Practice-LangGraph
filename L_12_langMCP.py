from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool, BaseTool
from langchain_groq import ChatGroq

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain.mcp import MCPAdapter

from dotenv import load_dotenv
from ddgs import DDGS

import aiosqlite
import asyncio
import threading


load_dotenv()


# ============================================================
# Dedicated async loop for backend tasks
# ============================================================

_ASYNC_LOOP = asyncio.new_event_loop()

_ASYNC_THREAD = threading.Thread(
    target=_ASYNC_LOOP.run_forever,
    daemon=True
)

_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(
        coro,
        _ASYNC_LOOP
    )


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# ============================================================
# 1. LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)


# ============================================================
# 2. Tools
# ============================================================

@tool
def search_tool(query: str) -> str:
    """
    Search the web for current information.

    Use this tool when answering questions that require
    up-to-date information, news, or facts from the web.

    Args:
        query: The search query.
    """

    try:

        with DDGS() as ddgs:

            results = list(
                ddgs.text(
                    query,
                    max_results=3
                )
            )

        if not results:
            return "No results found."

        return "\n\n".join(
            f"Title: {r.get('title', '')}\n"
            f"Snippet: {r.get('body', '')}\n"
            f"Source: {r.get('href', '')}"
            for r in results
        )

    except Exception as e:

        return (
            f"Search error: "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# 3. MCP configuration
# ============================================================

# ============================================================
# 3. MCP configuration
# ============================================================

mcp_config = {
    "mcpServers": {
        "arith": {
            "transport": "stdio",
            "command": "python3",
            "args": [
                "/home/omkar/Desktop/Pro/LG/Arithmcp.py"
            ],
        }
    }
}


# ============================================================
# 4. MCP Adapter
# ============================================================

_mcp_adapter = None


async def _init_mcp():

    global _mcp_adapter

    _mcp_adapter = MCPAdapter(
        mcp_config
    )

    await _mcp_adapter.__aenter__()

    tools = await _mcp_adapter.list_tools()

    print(
        f"Loaded {len(tools)} MCP tool(s)"
    )

    for mcp_tool in tools:
        print(
            f"  - {mcp_tool.name}"
        )

    return tools


def load_mcp_tools() -> list[BaseTool]:

    return run_async(
        _init_mcp()
    )


mcp_tools = load_mcp_tools()



# ============================================================
# 5. Combine all tools
# ============================================================

tools = [
    search_tool,
    *mcp_tools,
]


print(
    f"Total tools available: {len(tools)}"
)

for current_tool in tools:

    print(
        f"  - {current_tool.name}"
    )


llm_with_tools = (
    llm.bind_tools(tools)
    if tools
    else llm
)


# ============================================================
# 6. State
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# 7. Nodes
# ============================================================

async def chat_node(state: ChatState):
    """
    LLM node that may answer directly
    or request a tool call.
    """

    messages = state["messages"]

    response = await llm_with_tools.ainvoke(
        messages
    )

    return {
        "messages": [response]
    }


# ============================================================
# 8. Tool Node
# ============================================================

tool_node = (
    ToolNode(tools)
    if tools
    else None
)


# ============================================================
# 9. Async SQLite Checkpointer
# ============================================================

async def _init_checkpointer():

    # SQLite database file
    conn = await aiosqlite.connect(
        "checkpoints.sqlite"
    )

    # LangGraph async SQLite checkpointer
    checkpointer = AsyncSqliteSaver(
        conn
    )

    # Create required LangGraph tables
    await checkpointer.setup()

    return checkpointer


checkpointer = run_async(
    _init_checkpointer()
)


# ============================================================
# 10. Graph
# ============================================================

graph = StateGraph(
    ChatState
)


graph.add_node(
    "chat_node",
    chat_node
)


graph.add_edge(
    START,
    "chat_node"
)


if tool_node:

    graph.add_node(
        "tools",
        tool_node
    )

    graph.add_conditional_edges(
        "chat_node",
        tools_condition
    )

    graph.add_edge(
        "tools",
        "chat_node"
    )

else:

    graph.add_edge(
        "chat_node",
        END
    )


chatbot = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# 11. Retrieve all conversation threads
# ============================================================

async def _alist_threads():

    all_threads = set()

    async for checkpoint in checkpointer.alist(None):

        thread_id = (
            checkpoint.config
            .get("configurable", {})
            .get("thread_id")
        )

        if thread_id:

            all_threads.add(
                thread_id
            )

    return list(all_threads)


def retrieve_all_threads():

    return run_async(
        _alist_threads()
    )
