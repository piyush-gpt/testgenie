import streamlit as st
import os
from dotenv import load_dotenv
from loaders.api_loader import load_openapi_spec, chunk_spec_text
from vectorstore.store import build_vectorstore, get_retriever
from chains.qa_chain import build_qa_chain
import tempfile

load_dotenv()

st.set_page_config(
    page_title="TestGenie - API Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-family: monospace;
    }
    .stButton button {
        width: 100%;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #475063;
    }
    .chat-message .content {
        display: flex;
    }
    .chat-message .avatar {
        width: 2rem;
        height: 2rem;
        border-radius: 50%;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex: 1;
    }
    </style>
""", unsafe_allow_html=True)

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
    st.session_state.project_name = None
    st.session_state.messages = []

st.title("🤖 TestGenie")
st.markdown("### Your AI API Assistant")
st.markdown("Ask questions about your API or generate test cases")

with st.sidebar:
    st.header("About")
    st.markdown("""
    TestGenie is your AI assistant for API documentation and testing.
    
    **Features:**
    - Ask questions about API endpoints
    - Generate comprehensive test cases
    - Get detailed API explanations
    - Chat with context awareness
    """)
    
    st.header("Instructions")
    st.markdown("""
    1. Enter a project name (new or existing)
    2. Upload your OpenAPI/Swagger specification file
    3. Start chatting with TestGenie
    4. Ask about endpoints or request test cases
    """)

st.header("📁 Project Setup")
project_name = st.text_input(
    "Enter Project Name",
    placeholder="Enter a new project name or use an existing one",
    help="Use a new name for a new API documentation or an existing name to update/reuse previous documentation"
)

if not project_name:
    st.warning("⚠️ Please enter a project name to continue")
    st.stop()

st.header("📄 API Documentation")
uploaded_file = st.file_uploader("Upload your OpenAPI/Swagger specification file (YAML/JSON)", type=['yaml', 'yml', 'json'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        spec_text = load_openapi_spec(tmp_file_path)
        chunks = chunk_spec_text(spec_text)
        
        try:
            build_vectorstore(chunks, project_name)
        except Exception as e:
            st.error(f"❌ Error building vector store: {str(e)}")
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            st.stop()
        
        # Create new QA chain
        retriever = get_retriever(project_name)
        st.session_state.qa_chain = build_qa_chain(retriever)
        st.session_state.project_name = project_name
        
        st.success(f"✅ API documentation loaded successfully for project: {project_name}")
        
        os.unlink(tmp_file_path)
        
    except Exception as e:
        st.error(f"❌ Error loading API documentation: {str(e)}")
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
else:
    # Try to use existing project if no file is uploaded
    try:
        retriever = get_retriever(project_name)
        st.session_state.qa_chain = build_qa_chain(retriever)
        st.session_state.project_name = project_name
        st.info(f"ℹ️ Using existing API documentation for project: {project_name}")
    except ValueError as e:
        st.warning(str(e))
        st.warning("Please upload an API specification file to create a new project")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error accessing project: {str(e)}")
        st.stop()

st.header("💬 Chat with TestGenie")

for message in st.session_state.messages:
    with st.container():
        if message['role'] == 'assistant':
            st.markdown("""
            <div class="chat-message assistant">
                <div class="content">
                    <div class="avatar">🤖</div>
                    <div class="message">
            """, unsafe_allow_html=True)
            
            st.markdown(message['content'], unsafe_allow_html=False)
            
            st.markdown("""
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # For user messages, use simple text display
            st.markdown(f"""
            <div class="chat-message user">
                <div class="content">
                    <div class="avatar">👤</div>
                    <div class="message">{message['content']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

user_input = st.text_area(
    "Ask about your API or request test cases",
    placeholder="Example: Explain the /users POST endpoint or Generate test cases for user creation",
    height=100
)

if st.button("Send", type="primary"):
    if not st.session_state.qa_chain:
        st.error("❌ Please upload an API specification file first")
    elif not user_input:
        st.error("❌ Please enter your question")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.qa_chain.invoke({"input": user_input})
                
                st.session_state.messages.append({"role": "assistant", "content": response["output"]})
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.markdown("Built with ❤️ by TestGenie") 