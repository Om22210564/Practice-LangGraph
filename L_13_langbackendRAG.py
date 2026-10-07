from __future__ import annotations

import os
import tempfile
import asyncio
import threading
import aiosqlite

from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from ddgs import DDGS

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain_core.tools import (
    tool,
    BaseTool,
    InjectedToolArg,
)
from langchain_core.runnables import RunnableConfig

from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import HuggingFaceEmbeddings

from langchain.mcp import MCPAdapter

from langgraph.graph import (
    StateGraph,
    START,
)

from langgraph.graph.message import add_messages

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from langgraph.checkpoint.sqlite.aio import (
    AsyncSqliteSaver
)


load_dotenv()


# ============================================================
# 1. Embeddings
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 2. Dedicated async loop
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
    """
    Schedule a coroutine on the backend
    event loop.
    """

    return _submit_async(coro)


# ============================================================
# 3. LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)


# ============================================================
# 4. RAG storage
# ============================================================

# Each conversation thread gets its own retriever.
#
# thread_1 -> PDF A -> FAISS retriever
# thread_2 -> PDF B -> FAISS retriever
#

_THREAD_RETRIEVERS: Dict[str, Any] = {}

_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(
    thread_id: Optional[str]
):

    if not thread_id:
        return None

    return _THREAD_RETRIEVERS.get(
        str(thread_id)
    )


# ============================================================
# 5. PDF ingestion
# ============================================================

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None
) -> dict:
    """
    Load a PDF, split it into chunks, create
    a FAISS vector store and associate the
    retriever with the current thread.
    """

    if not file_bytes:

        raise ValueError(
            "No bytes received for ingestion."
        )


    thread_id = str(thread_id)


    # --------------------------------------------------------
    # Create temporary PDF
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(
            file_bytes
        )

        temp_path = temp_file.name


    try:

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        loader = PyPDFLoader(
            temp_path
        )

        documents = loader.load()


        if not documents:

            raise ValueError(
                "The PDF contains no readable content."
            )


        # ----------------------------------------------------
        # Split documents
        # ----------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )


        chunks = splitter.split_documents(
            documents
        )


        if not chunks:

            raise ValueError(
                "Could not create text chunks from PDF."
            )


        # ----------------------------------------------------
        # Create FAISS vector store
        # ----------------------------------------------------

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )


        # ----------------------------------------------------
        # Create retriever
        # ----------------------------------------------------

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4
            }
        )


        # ----------------------------------------------------
        # Store retriever
        # ----------------------------------------------------

        _THREAD_RETRIEVERS[
            thread_id
        ] = retriever


        _THREAD_METADATA[
            thread_id
        ] = {

            "filename": (
                filename
                or os.path.basename(temp_path)
            ),

            "documents": len(documents),

            "chunks": len(chunks),
        }


        return {

            "thread_id": thread_id,

            "filename": (
                filename
                or os.path.basename(temp_path)
            ),

            "documents": len(documents),

            "chunks": len(chunks),
        }


    finally:

        try:

            os.remove(
                temp_path
            )

        except OSError:

            pass


# ============================================================
# 6. Web search tool
# ============================================================

@tool
def search_tool(
    query: str
) -> str:
    """
    Search the web for current information.

    Use this tool for up-to-date information,
    news and current facts.
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
# 7. RAG tool
# ============================================================

@tool
def rag_tool(
    query: str,
    config: Annotated[
        RunnableConfig,
        InjectedToolArg
    ],
) -> dict:
    """
    Retrieve relevant information from the PDF
    uploaded to the current chat thread.

    The thread_id is injected automatically from
    LangGraph configuration.

    The LLM only needs to provide the query.
    """

    # --------------------------------------------------------
    # Get thread_id from LangGraph config
    # --------------------------------------------------------

    thread_id = (
        config
        .get("configurable", {})
        .get("thread_id")
    )


    if not thread_id:

        return {

            "error": (
                "No thread_id was found in "
                "the LangGraph configuration."
            ),

            "query": query,
        }


    thread_id = str(
        thread_id
    )


    # --------------------------------------------------------
    # Get retriever
    # --------------------------------------------------------

    retriever = _get_retriever(
        thread_id
    )


    if retriever is None:

        return {

            "error": (
                "No PDF has been indexed for "
                "this chat thread. "
                "Please upload a PDF first."
            ),

            "query": query,

            "thread_id": thread_id,
        }


    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    try:

        documents = retriever.invoke(
            query
        )


        if not documents:

            return {

                "query": query,

                "context": [],

                "metadata": [],

                "source_file": (
                    _THREAD_METADATA
                    .get(thread_id, {})
                    .get("filename")
                ),
            }


        context = [
            document.page_content
            for document in documents
        ]


        metadata = [
            document.metadata
            for document in documents
        ]


        return {

            "query": query,

            "context": context,

            "metadata": metadata,

            "source_file": (
                _THREAD_METADATA
                .get(thread_id, {})
                .get("filename")
            ),
        }


    except Exception as e:

        return {

            "error": (
                f"RAG retrieval failed: "
                f"{type(e).__name__}: {e}"
            ),

            "query": query,

            "thread_id": thread_id,
        }


# ============================================================
# 8. MCP configuration
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
# 9. MCP Adapter
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
# 10. Combine tools
# ============================================================

tools = [

    search_tool,

    rag_tool,

    *mcp_tools,

]


print(
    f"Total tools available: {len(tools)}"
)


for current_tool in tools:

    print(
        f"  - {current_tool.name}"
    )


# ============================================================
# 11. Bind tools to LLM
# ============================================================

llm_with_tools = llm.bind_tools(
    tools
)


# ============================================================
# 12. State
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# 13. Chat node
# ============================================================

async def chat_node(
    state: ChatState,
    config=None
):
    """
    LLM node that can answer directly
    or request a tool call.
    """

    # --------------------------------------------------------
    # Get current thread ID
    # --------------------------------------------------------

    thread_id = None


    if config and isinstance(
        config,
        dict
    ):

        thread_id = (
            config
            .get("configurable", {})
            .get("thread_id")
        )


    # --------------------------------------------------------
    # System instructions
    # --------------------------------------------------------

    system_message = SystemMessage(

        content=(

            "You are a helpful AI assistant.\n\n"

            "You have access to these tools:\n\n"

            "1. rag_tool\n"
            "Use rag_tool when the user asks "
            "about an uploaded PDF or document.\n"
            "The tool automatically knows the "
            "current conversation thread.\n\n"

            "2. search_tool\n"
            "Use search_tool for current information, "
            "news, recent facts, or web information.\n\n"

            "3. MCP tools\n"
            "Use MCP tools when the user's request "
            "matches their functionality.\n\n"

            "IMPORTANT:\n"
            "If the user asks about the uploaded PDF, "
            "prefer rag_tool.\n\n"

            "Do NOT provide a thread_id to rag_tool. "
            "The application supplies it automatically.\n\n"

            "When answering from the PDF, use the "
            "retrieved context and do not invent "
            "information that is not supported by it."
        )
    )


    messages = [

        system_message,

        *state["messages"],

    ]


    try:

        response = await llm_with_tools.ainvoke(

            messages,

            config=config

        )


    except Exception as e:

        print(
            "\n========== LLM ERROR =========="
        )

        print(
            type(e).__name__
        )

        print(
            str(e)
        )

        print(
            "================================\n"
        )

        raise


    return {

        "messages": [
            response
        ]

    }


# ============================================================
# 14. Tool node
# ============================================================

tool_node = ToolNode(
    tools
)


# ============================================================
# 15. Async SQLite Checkpointer
# ============================================================

async def _init_checkpointer():

    conn = await aiosqlite.connect(
        "checkpoints.sqlite"
    )


    checkpointer = AsyncSqliteSaver(
        conn
    )


    await checkpointer.setup()


    return checkpointer


checkpointer = run_async(
    _init_checkpointer()
)


# ============================================================
# 16. Graph
# ============================================================

graph = StateGraph(
    ChatState
)


graph.add_node(
    "chat_node",
    chat_node
)


graph.add_node(
    "tools",
    tool_node
)


graph.add_edge(
    START,
    "chat_node"
)


graph.add_conditional_edges(
    "chat_node",
    tools_condition
)


graph.add_edge(
    "tools",
    "chat_node"
)


# ============================================================
# 17. Compile
# ============================================================

chatbot = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# 18. Retrieve all conversation threads
# ============================================================

async def _alist_threads():

    all_threads = set()


    async for checkpoint in (
        checkpointer.alist(None)
    ):

        thread_id = (

            checkpoint.config
            .get("configurable", {})
            .get("thread_id")

        )


        if thread_id:

            all_threads.add(
                thread_id
            )


    return list(
        all_threads
    )


def retrieve_all_threads():

    return run_async(
        _alist_threads()
    )


# ============================================================
# 19. RAG helper functions
# ============================================================

def thread_has_document(
    thread_id: str
) -> bool:
    """
    Check whether a PDF has been indexed
    for a conversation thread.
    """

    return (
        str(thread_id)
        in _THREAD_RETRIEVERS
    )


def thread_document_metadata(
    thread_id: str
) -> dict:
    """
    Get metadata for the PDF associated
    with a conversation thread.
    """

    return _THREAD_METADATA.get(
        str(thread_id),
        {}
    )


def remove_thread_document(
    thread_id: str
) -> bool:
    """
    Remove the in-memory retriever
    associated with a thread.
    """

    thread_id = str(
        thread_id
    )


    removed = False


    if thread_id in _THREAD_RETRIEVERS:

        del _THREAD_RETRIEVERS[
            thread_id
        ]

        removed = True


    if thread_id in _THREAD_METADATA:

        del _THREAD_METADATA[
            thread_id
        ]


    return removed