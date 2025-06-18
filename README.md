# TestGenie 🤖

**TestGenie** is an AI-powered assistant for API documentation and automated test case generation. Built with Streamlit, LangChain, OpenAI, and FAISS, it helps developers and QA engineers quickly understand API endpoints and generate comprehensive pytest test cases from OpenAPI/Swagger specifications.

---

## 🚀 Features

- **AI-Powered Test Generation:** Automatically generate happy path, edge case, and error case test scenarios for your API endpoints using GPT models.
- **API Endpoint Explanation:** Get clear, detailed explanations of API endpoints, including request/response examples and authentication requirements.
- **Conversational Interface:** Chat with the assistant to ask questions about your API or request test cases, with context-aware memory.
- **Project Management:** Upload and manage multiple API documentation projects.
- **Fast Vector Search:** Uses FAISS for efficient document retrieval and semantic search.

---

## 🖥️ Demo

![TestGenie Demo](demo.gif)

---

## 📦 Installation

### Prerequisites
- Python 3.13+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/testgenie.git
cd testgenie
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your OpenAI API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-openai-api-key
```

---

## 🏃‍♂️ Usage

```bash
streamlit run app.py
```

1. Enter a project name (new or existing)
2. Upload your OpenAPI/Swagger specification file (YAML/JSON)
3. Start chatting with TestGenie to generate test cases or get API explanations

---

## 📚 Example Queries
- "Generate test cases for the /users POST endpoint"
- "Explain the /auth/login endpoint"
- "What are the possible error responses for /orders?"

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) - UI
- [LangChain](https://python.langchain.com/) - LLM Orchestration
- [OpenAI GPT](https://platform.openai.com/docs/models) - Language Model
- [FAISS](https://github.com/facebookresearch/faiss) - Vector Search

---



## 🙏 Acknowledgements
- [OpenAI](https://openai.com/)
- [LangChain](https://langchain.com/)
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss) 
