from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

# Test Generation Prompt
TEST_GENERATION_PROMPT = """
You are an expert QA engineer specializing in API testing. Your task is to generate comprehensive test cases based on the API documentation and user requirements.

Context from API Documentation:
{context}

User Question:
{question}

Please generate test cases following these guidelines:
1. Include test cases for:
   - Happy path scenarios
   - Edge cases
   - Error cases
   - Input validation
   - Response validation
2. Use pytest framework
3. Include proper assertions and error handling
4. Add descriptive test names and comments
5. Include setup and teardown if needed
6. Consider rate limiting, authentication, and security if applicable

IMPORTANT: Format your response with proper markdown code blocks. Use triple backticks with 'python' language specification.
Example:
```python
# Your test code here
```

"""

# API Explanation Prompt
API_EXPLANATION_PROMPT = """
You are an expert API documentation specialist. Your task is to explain API endpoints in a clear and comprehensive way.

Context from API Documentation:
{context}

User Question:
{question}

Please provide a detailed explanation following these guidelines:
1. Explain the endpoint's purpose and functionality
2. Describe the request parameters and body structure
3. Explain the possible response codes and their meanings
4. Provide examples of request and response
5. Mention any authentication requirements
6. Include any important notes or limitations


Format the response in a clear, conversational way that's easy to understand.

"""

def build_qa_chain(retriever):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)
    
    test_prompt = ChatPromptTemplate.from_template(TEST_GENERATION_PROMPT)
    api_prompt = ChatPromptTemplate.from_template(API_EXPLANATION_PROMPT)
    
    # Define tools
    def generate_tests(question: str) -> str:
        """Generate test cases for API endpoints."""
        context = retriever.get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in context])
        response = llm.invoke(
            test_prompt.format(
                context=context_text,
                question=question
            )
        )
       
        return response.content

    def explain_api(question: str) -> str:
        """Explain API endpoints and their functionality."""
        context = retriever.get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in context])
        response = llm.invoke(
            api_prompt.format(
                context=context_text,
                question=question
            )
        )
        return response.content

    tools = [
        Tool(
            name="generate_tests",
            func=generate_tests,
            description="Use this tool when you need to generate test cases for API endpoints. Input should be a question about generating tests."
        ),
        Tool(
            name="explain_api",
            func=explain_api,
            description="Use this tool when you need to understand or explain API endpoints. Input should be a question about API functionality."
        )
    ]

    # Create the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant that helps generate test cases and explain API endpoints.
Your ONLY job is to:
1. Use the appropriate tool (generate_tests or explain_api)
2. Return the EXACT output from the tool without ANY modifications
3. Do not add any commentary, summaries, or explanations
4. Do not wrap the output in any additional formatting

IMPORTANT: 
- NEVER summarize or modify the tool's output
- Return the tool's output exactly as received
- Do not add any additional text or formatting"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor
