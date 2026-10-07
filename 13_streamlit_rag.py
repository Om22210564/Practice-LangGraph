import queue
import uuid

import streamlit as st

from L_13_langbackendRAG import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    submit_async_task,
    thread_document_metadata,
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)


# ============================================================
# Utilities
# ============================================================

def generate_thread_id():
    return uuid.uuid4()


def reset_chat():

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id

    add_thread(thread_id)

    st.session_state["message_history"] = []


def add_thread(thread_id):

    if thread_id not in st.session_state["chat_threads"]:

        st.session_state["chat_threads"].append(
            thread_id
        )


def load_conversation(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get(
        "messages",
        []
    )


# ============================================================
# Session Initialization
# ============================================================

if "message_history" not in st.session_state:

    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = (
        generate_thread_id()
    )


if "chat_threads" not in st.session_state:

    st.session_state["chat_threads"] = (
        retrieve_all_threads()
    )


# Track PDFs indexed for each thread
if "ingested_docs" not in st.session_state:

    st.session_state["ingested_docs"] = {}


add_thread(
    st.session_state["thread_id"]
)


# Current thread
thread_key = str(
    st.session_state["thread_id"]
)


# Documents associated with current thread
thread_docs = (
    st.session_state["ingested_docs"]
    .setdefault(thread_key, {})
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title(
    "LangGraph MCP Chatbot"
)


# Show current thread
st.sidebar.markdown(
    f"**Thread ID:** `{thread_key}`"
)


# ------------------------------------------------------------
# New Chat
# ------------------------------------------------------------

if st.sidebar.button(
    "New Chat",
    use_container_width=True
):

    reset_chat()

    st.rerun()


# ============================================================
# PDF / RAG Section
# ============================================================

st.sidebar.subheader(
    "📄 PDF Knowledge Base"
)


# Show currently indexed document
if thread_docs:

    latest_doc = list(
        thread_docs.values()
    )[-1]

    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from "
        f"{latest_doc.get('documents')} pages)"
    )

else:

    st.sidebar.info(
        "No PDF indexed for this chat."
    )


# ------------------------------------------------------------
# PDF uploader
# ------------------------------------------------------------

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat",
    type=["pdf"]
)


if uploaded_pdf:

    # Avoid indexing the same filename repeatedly
    if uploaded_pdf.name in thread_docs:

        st.sidebar.info(
            f"`{uploaded_pdf.name}` "
            "is already indexed for this chat."
        )

    else:

        with st.sidebar.status(
            "Indexing PDF…",
            expanded=True
        ) as status_box:

            try:

                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )

                thread_docs[
                    uploaded_pdf.name
                ] = summary

                status_box.update(
                    label="✅ PDF indexed",
                    state="complete",
                    expanded=False
                )

            except Exception as exc:

                status_box.update(
                    label="❌ PDF indexing failed",
                    state="error",
                    expanded=True
                )

                st.sidebar.error(
                    str(exc)
                )


# ============================================================
# Conversations
# ============================================================

st.sidebar.subheader(
    "My Conversations"
)


selected_thread = None


threads = (
    st.session_state["chat_threads"][::-1]
)


if not threads:

    st.sidebar.write(
        "No past conversations yet."
    )

else:

    for thread_id in threads:

        if st.sidebar.button(
            str(thread_id),
            key=f"side-thread-{thread_id}"
        ):

            selected_thread = thread_id


# ============================================================
# Main UI
# ============================================================

st.title(
    "Multi Utility Chatbot"
)


# ============================================================
# Render conversation history
# ============================================================

for message in (
    st.session_state["message_history"]
):

    with st.chat_message(
        message["role"]
    ):

        st.text(
            message["content"]
        )


# ============================================================
# Chat input
# ============================================================

user_input = st.chat_input(
    "Ask about your document or use tools"
)


if user_input:

    # --------------------------------------------------------
    # Show user message
    # --------------------------------------------------------

    st.session_state[
        "message_history"
    ].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):

        st.text(
            user_input
        )


    # --------------------------------------------------------
    # LangGraph configuration
    # --------------------------------------------------------

    CONFIG = {

        "configurable": {
            "thread_id": thread_key
        },

        "metadata": {
            "thread_id": thread_key
        },

        "run_name": "chat_turn",
    }


    # --------------------------------------------------------
    # Assistant response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        status_holder = {
            "box": None
        }


        def ai_only_stream():

            event_queue: queue.Queue = (
                queue.Queue()
            )


            async def run_stream():

                try:

                    async for (
                        message_chunk,
                        metadata
                    ) in chatbot.astream(

                        {
                            "messages": [
                                HumanMessage(
                                    content=user_input
                                )
                            ]
                        },

                        config=CONFIG,

                        stream_mode="messages",
                    ):

                        event_queue.put(
                            (
                                message_chunk,
                                metadata
                            )
                        )


                except Exception as exc:

                    event_queue.put(
                        (
                            "error",
                            exc
                        )
                    )


                finally:

                    event_queue.put(
                        None
                    )


            # Run LangGraph on the backend
            # async event loop
            submit_async_task(
                run_stream()
            )


            # Consume streaming events
            while True:

                item = event_queue.get()


                if item is None:

                    break


                message_chunk, metadata = item


                if message_chunk == "error":

                    raise metadata


                # ------------------------------------------------
                # Tool execution indicator
                # ------------------------------------------------

                if isinstance(
                    message_chunk,
                    ToolMessage
                ):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )


                    if status_holder[
                        "box"
                    ] is None:

                        status_holder[
                            "box"
                        ] = st.status(

                            f"🔧 Using `{tool_name}` …",

                            expanded=True
                        )

                    else:

                        status_holder[
                            "box"
                        ].update(

                            label=(
                                f"🔧 Using "
                                f"`{tool_name}` …"
                            ),

                            state="running",

                            expanded=True
                        )


                # ------------------------------------------------
                # Stream only assistant output
                # ------------------------------------------------

                if isinstance(
                    message_chunk,
                    AIMessage
                ):

                    # AIMessage.content can occasionally
                    # be a non-string structure.
                    if isinstance(
                        message_chunk.content,
                        str
                    ):

                        yield (
                            message_chunk.content
                        )


        # Stream response to UI
        ai_message = st.write_stream(
            ai_only_stream()
        )


        # --------------------------------------------------------
        # Finalize tool status
        # --------------------------------------------------------

        if status_holder[
            "box"
        ] is not None:

            status_holder[
                "box"
            ].update(

                label="✅ Tool finished",

                state="complete",

                expanded=False
            )


    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    st.session_state[
        "message_history"
    ].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )


    # --------------------------------------------------------
    # Show document information
    # --------------------------------------------------------

    doc_meta = thread_document_metadata(
        thread_key
    )


    if doc_meta:

        st.caption(

            f"📄 Document indexed: "
            f"{doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, "
            f"pages: {doc_meta.get('documents')})"

        )


# ============================================================
# Load selected conversation
# ============================================================

if selected_thread:

    st.session_state[
        "thread_id"
    ] = selected_thread


    messages = load_conversation(
        selected_thread
    )


    temp_messages = []


    for msg in messages:

        # Ignore tool messages when rebuilding
        # visible chat history.
        if isinstance(
            msg,
            HumanMessage
        ):

            role = "user"

        elif isinstance(
            msg,
            AIMessage
        ):

            role = "assistant"

        else:

            continue


        # AI messages can occasionally have
        # non-string content.
        if isinstance(
            msg.content,
            str
        ):

            temp_messages.append(
                {
                    "role": role,
                    "content": msg.content
                }
            )


    st.session_state[
        "message_history"
    ] = temp_messages


    # Make sure the selected thread has
    # a document tracking dictionary.
    st.session_state[
        "ingested_docs"
    ].setdefault(
        str(selected_thread),
        {}
    )


    st.rerun()