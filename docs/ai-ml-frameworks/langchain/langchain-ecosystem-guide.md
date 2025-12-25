# LangChain Ecosystem Tools Guide

A comprehensive guide to LangFlow, LangSmith, Langfuse, and LangGraph - essential tools for building, monitoring, and optimizing LLM applications.

---

## Table of Contents

1. [LangFlow](#langflow)
2. [LangSmith](#langsmith)
3. [Langfuse](#langfuse)
4. [LangGraph](#langgraph)

---

# LangFlow

## What is LangFlow?

LangFlow is a visual, no-code/low-code tool for building LangChain applications. It provides a drag-and-drop interface to create complex AI workflows without writing extensive code.

## Installation

### Using pip

```bash
pip install langflow
```

### Using Docker

```bash
docker run -it -p 7860:7860 langflowai/langflow:latest
```

### From Source

```bash
git clone https://github.com/logspace-ai/langflow.git
cd langflow
pip install -e .
```

## Starting LangFlow

```bash
# Start LangFlow
langflow run

# Specify host and port
langflow run --host 0.0.0.0 --port 7860

# With custom backend
langflow run --backend-only
```

Access at: `http://localhost:7860`

## Core Concepts

### Components

1. **LLMs** - Language models (OpenAI, Anthropic, etc.)
2. **Prompts** - Prompt templates
3. **Chains** - Sequence of operations
4. **Agents** - Autonomous decision-making
5. **Tools** - External functionalities
6. **Memory** - Conversation history
7. **Document Loaders** - Data ingestion
8. **Vector Stores** - Embedding storage

## Building Your First Flow

### 1. Simple Chat Application

```
Components:
- ChatOpenAI (LLM)
- ConversationBufferMemory
- ConversationChain
- ChatInput
- ChatOutput
```

**Steps:**
1. Drag "ChatOpenAI" to canvas
2. Add "ConversationBufferMemory"
3. Add "ConversationChain"
4. Connect components
5. Add API keys in component settings
6. Test with the chat interface

### 2. RAG (Retrieval-Augmented Generation)

```
Flow:
Document Loader → Text Splitter → Embeddings → 
Vector Store → Retriever → QA Chain → Output
```

**Example Components:**
- PDFLoader
- RecursiveCharacterTextSplitter
- OpenAIEmbeddings
- Chroma/Pinecone
- RetrievalQA

### 3. Agent with Tools

```
Agent Flow:
Tools (Wikipedia, Calculator, Search) → 
Agent → LLM → Output
```

## Custom Components

### Python Custom Component

```python
from langflow import CustomComponent
from langchain.schema import Document

class MyCustomComponent(CustomComponent):
    display_name = "My Component"
    description = "Does something custom"
    
    def build_config(self):
        return {
            "input_text": {
                "display_name": "Input Text",
                "type": "str"
            }
        }
    
    def build(self, input_text: str) -> Document:
        # Your logic here
        processed = input_text.upper()
        return Document(page_content=processed)
```

## API Usage

### Export Flow

```bash
# Export from UI: Flow Settings → Export → JSON
```

### Run via API

```python
import requests

url = "http://localhost:7860/api/v1/process"
payload = {
    "flow_id": "your-flow-id",
    "inputs": {"input": "Hello, world!"}
}

response = requests.post(url, json=payload)
print(response.json())
```

## Environment Variables

```bash
# OpenAI
OPENAI_API_KEY=your-key

# Anthropic
ANTHROPIC_API_KEY=your-key

# Database
LANGFLOW_DATABASE_URL=sqlite:///./langflow.db

# Authentication
LANGFLOW_AUTO_LOGIN=false
```

## Best Practices

1. **Modular Design** - Break complex flows into reusable components
2. **Version Control** - Export and save flows regularly
3. **Test Incrementally** - Test each component individually
4. **Use Templates** - Start with pre-built templates
5. **Document Flows** - Add descriptions to components

---

# LangSmith

## What is LangSmith?

LangSmith is LangChain's official platform for debugging, testing, evaluating, and monitoring LLM applications in production.

## Setup

### Installation

```bash
pip install langsmith langchain
```

### Configuration

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "your-project-name"
```

Get API key from: https://smith.langchain.com

## Core Features

### 1. Tracing

Automatically trace all LangChain operations:

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Tracing is automatic when env vars are set
llm = ChatOpenAI(temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

chain = prompt | llm
result = chain.invoke({"input": "What is LangSmith?"})
```

### 2. Custom Tracing

```python
from langsmith import traceable

@traceable(run_type="chain", name="My Custom Chain")
def my_function(text: str) -> str:
    # Your logic
    return text.upper()

result = my_function("hello")
```

### 3. Adding Metadata

```python
from langsmith import Client

client = Client()

# Add tags and metadata
chain.invoke(
    {"input": "test"},
    config={
        "tags": ["production", "user-123"],
        "metadata": {"user_id": "123", "session": "abc"}
    }
)
```

## Datasets and Testing

### Create Dataset

```python
from langsmith import Client

client = Client()

# Create dataset
dataset = client.create_dataset("my-test-dataset")

# Add examples
client.create_examples(
    inputs=[
        {"input": "What is AI?"},
        {"input": "Explain machine learning"}
    ],
    outputs=[
        {"output": "Artificial Intelligence is..."},
        {"output": "Machine learning is..."}
    ],
    dataset_id=dataset.id
)
```

### Run Evaluation

```python
from langsmith.evaluation import evaluate

# Define your chain
def my_chain(inputs: dict) -> dict:
    return chain.invoke(inputs)

# Run evaluation
results = evaluate(
    my_chain,
    data="my-test-dataset",
    evaluators=[...],  # Custom evaluators
    experiment_prefix="test-run"
)
```

## Custom Evaluators

```python
from langsmith.evaluation import EvaluationResult

def accuracy_evaluator(run, example):
    predicted = run.outputs["output"]
    expected = example.outputs["output"]
    
    score = 1.0 if predicted == expected else 0.0
    
    return EvaluationResult(
        key="exact_match",
        score=score
    )

# Use in evaluation
results = evaluate(
    my_chain,
    data="my-test-dataset",
    evaluators=[accuracy_evaluator]
)
```

## Monitoring

### View Traces

1. Go to https://smith.langchain.com
2. Select your project
3. View traces with latency, cost, and errors

### Filtering

```python
from langsmith import Client

client = Client()

# Get runs with filters
runs = client.list_runs(
    project_name="my-project",
    filter='eq(status, "error")',
    limit=100
)

for run in runs:
    print(f"Error: {run.error}")
```

## Feedback and Annotations

```python
from langsmith import Client

client = Client()

# Add feedback to a run
client.create_feedback(
    run_id="run-id",
    key="user_rating",
    score=0.9,
    comment="Great response!"
)
```

## Best Practices

1. **Use Projects** - Organize by environment (dev, staging, prod)
2. **Tag Runs** - Add meaningful tags for filtering
3. **Create Datasets** - Build test suites for regression testing
4. **Monitor Costs** - Track token usage and costs
5. **Set Alerts** - Configure alerts for errors or high latency

---

# Langfuse

## What is Langfuse?

Langfuse is an open-source LLM engineering platform for debugging, analyzing, and iterating on LLM applications. It's self-hostable and privacy-focused.

## Installation

### Cloud Version

Sign up at: https://cloud.langfuse.com

### Self-Hosted (Docker)

```bash
# Clone repository
git clone https://github.com/langfuse/langfuse.git
cd langfuse

# Start with Docker Compose
docker-compose up -d
```

### Python SDK

```bash
pip install langfuse
```

## Setup

```python
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="your-public-key",
    secret_key="your-secret-key",
    host="https://cloud.langfuse.com"  # or your self-hosted URL
)
```

## Core Features

### 1. Tracing

```python
from langfuse.decorators import observe, langfuse_context

@observe()
def generate_response(user_input: str) -> str:
    # Your LLM logic
    response = llm.invoke(user_input)
    return response

@observe()
def main():
    result = generate_response("What is AI?")
    return result

# Set user ID
langfuse_context.update_current_trace(
    user_id="user-123",
    session_id="session-456"
)
```

### 2. Manual Tracing

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Create trace
trace = langfuse.trace(
    name="chat-completion",
    user_id="user-123",
    metadata={"environment": "production"}
)

# Add generation
generation = trace.generation(
    name="openai-call",
    model="gpt-4",
    input={"messages": [{"role": "user", "content": "Hello"}]},
    output={"content": "Hi there!"},
    usage={
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15
    }
)
```

### 3. LangChain Integration

```python
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    public_key="your-public-key",
    secret_key="your-secret-key"
)

# Use with LangChain
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI()
response = llm.invoke(
    "Hello!",
    config={"callbacks": [langfuse_handler]}
)
```

## Prompt Management

### Create Prompt Template

```python
# Create prompt in Langfuse UI or via API
langfuse.create_prompt(
    name="chat-prompt",
    prompt="You are a helpful assistant. User: {{input}}",
    config={
        "model": "gpt-4",
        "temperature": 0.7
    }
)
```

### Use Prompt Template

```python
# Fetch and use prompt
prompt = langfuse.get_prompt("chat-prompt")

# Use in your application
response = llm.invoke(
    prompt.compile(input="What is AI?")
)

# Link to trace
trace.generation(
    prompt=prompt
)
```

## Scoring and Evaluation

### Add Scores

```python
# Score a trace
langfuse.score(
    trace_id="trace-id",
    name="user-feedback",
    value=1,  # 0 or 1
    comment="Great response"
)

# Score with numeric value
langfuse.score(
    trace_id="trace-id",
    name="relevance",
    value=0.95,  # 0.0 to 1.0
    data_type="NUMERIC"
)
```

### Automated Evaluation

```python
from langfuse.decorators import observe

@observe()
def evaluate_response(trace_id: str, output: str):
    # Your evaluation logic
    score = calculate_quality(output)
    
    langfuse.score(
        trace_id=trace_id,
        name="quality",
        value=score
    )
```

## Datasets

### Create Dataset

```python
# Create dataset
dataset = langfuse.create_dataset(
    name="qa-dataset",
    description="Question answering test set"
)

# Add items
dataset.create_item(
    input={"question": "What is AI?"},
    expected_output={"answer": "Artificial Intelligence is..."}
)
```

### Run Evaluation

```python
dataset = langfuse.get_dataset("qa-dataset")

for item in dataset.items:
    # Run your model
    output = your_model(item.input)
    
    # Create observation
    langfuse.trace(
        name="evaluation",
        input=item.input,
        output=output,
        metadata={"dataset_item_id": item.id}
    )
```

## Analytics and Insights

### Token Usage

```python
# Automatic tracking with generations
trace.generation(
    model="gpt-4",
    usage={
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
    },
    metadata={
        "cost": 0.0045  # Optional cost tracking
    }
)
```

### Custom Metrics

```python
# Track custom metrics
trace = langfuse.trace(
    name="custom-workflow",
    metadata={
        "latency_ms": 1500,
        "cache_hit": True,
        "processing_time": 0.5
    }
)
```

## Best Practices

1. **Use Decorators** - Simplify tracing with `@observe()`
2. **Track User Sessions** - Link traces to users and sessions
3. **Version Prompts** - Use prompt management for versioning
4. **Add Metadata** - Include context for better analysis
5. **Score Regularly** - Collect feedback and scores
6. **Self-Host** - For sensitive data, use self-hosted version

---

# LangGraph

## What is LangGraph?

LangGraph is a library for building stateful, multi-agent applications with LLMs. It extends LangChain with graph-based workflows and advanced state management.

## Installation

```bash
pip install langgraph langchain
```

## Core Concepts

### 1. State Graph

A graph where nodes are functions and edges define the flow.

### 2. State

Shared state that persists across nodes.

### 3. Nodes

Functions that process state.

### 4. Edges

Connections between nodes (conditional or direct).

## Basic Example

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next: str

# Define nodes
def chatbot(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def router(state: AgentState):
    # Decide next step
    last_message = state["messages"][-1]
    if "bye" in last_message.content.lower():
        return "end"
    return "continue"

# Build graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("chatbot", chatbot)
workflow.add_node("router", router)

# Add edges
workflow.set_entry_point("chatbot")
workflow.add_conditional_edges(
    "chatbot",
    router,
    {
        "continue": "chatbot",
        "end": END
    }
)

# Compile
app = workflow.compile()

# Run
result = app.invoke({
    "messages": [HumanMessage(content="Hello!")]
})
```

## Agent with Tools

```python
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain.tools import Tool

# Define tools
def search(query: str) -> str:
    return f"Results for: {query}"

tools = [
    Tool(
        name="search",
        func=search,
        description="Search the web"
    )
]

tool_executor = ToolExecutor(tools)

# Agent function
def run_agent(state: AgentState):
    response = agent.invoke(state["messages"])
    return {"messages": [response]}

# Tool execution node
def execute_tools(state: AgentState):
    last_message = state["messages"][-1]
    tool_invocations = last_message.tool_calls
    
    results = []
    for invocation in tool_invocations:
        result = tool_executor.invoke(invocation)
        results.append(result)
    
    return {"messages": results}

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_node("tools", execute_tools)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "agent")

app = workflow.compile()
```

## Multi-Agent System

```python
from langgraph.graph import StateGraph

class MultiAgentState(TypedDict):
    messages: list
    current_agent: str
    task_complete: bool

# Define agents
def researcher(state: MultiAgentState):
    # Research agent logic
    research = perform_research(state["messages"])
    return {
        "messages": state["messages"] + [research],
        "current_agent": "writer"
    }

def writer(state: MultiAgentState):
    # Writer agent logic
    content = write_content(state["messages"])
    return {
        "messages": state["messages"] + [content],
        "current_agent": "reviewer"
    }

def reviewer(state: MultiAgentState):
    # Reviewer agent logic
    review = review_content(state["messages"])
    
    if review["approved"]:
        return {
            "messages": state["messages"] + [review],
            "task_complete": True
        }
    else:
        return {
            "messages": state["messages"] + [review],
            "current_agent": "writer"
        }

# Build graph
workflow = StateGraph(MultiAgentState)
workflow.add_node("researcher", researcher)
workflow.add_node("writer", writer)
workflow.add_node("reviewer", reviewer)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    lambda x: "end" if x["task_complete"] else "writer",
    {
        "writer": "writer",
        "end": END
    }
)

app = workflow.compile()
```

## Persistence and Checkpoints

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Add checkpointing
memory = SqliteSaver.from_conn_string("checkpoints.db")

app = workflow.compile(checkpointer=memory)

# Run with thread ID for persistence
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"messages": [...]}, config=config)

# Continue conversation
result = app.invoke({"messages": [...]}, config=config)
```

## Streaming

```python
# Stream tokens
for chunk in app.stream({"messages": [HumanMessage(content="Hello")]}):
    print(chunk)

# Stream with updates
for update in app.stream(
    {"messages": [...]},
    config=config,
    stream_mode="updates"
):
    print(update)
```

## Human-in-the-Loop

```python
from langgraph.prebuilt import create_react_agent

# Create agent with interrupt
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["tools"]  # Pause before tool execution
)

# Run until interrupt
result = app.invoke({"messages": [...]}, config=config)

# Review and approve
print("Tools to execute:", result)

# Continue after approval
result = app.invoke(None, config=config)
```

## Best Practices

1. **Design State Carefully** - Include all necessary information
2. **Use Conditional Edges** - Create dynamic workflows
3. **Add Checkpoints** - Enable conversation persistence
4. **Test Incrementally** - Test each node independently
5. **Handle Errors** - Add error handling in nodes
6. **Use Type Hints** - Define clear state types
7. **Monitor Performance** - Track graph execution time

## Advanced Patterns

### Parallel Processing

```python
from langgraph.graph import StateGraph

# Multiple parallel branches
workflow.add_node("task1", task1_func)
workflow.add_node("task2", task2_func)
workflow.add_node("merge", merge_func)

workflow.add_edge("start", "task1")
workflow.add_edge("start", "task2")
workflow.add_edge("task1", "merge")
workflow.add_edge("task2", "merge")
```

### Subgraphs

```python
# Create subgraph
subgraph = StateGraph(SubState)
subgraph.add_node("step1", step1)
subgraph.add_node("step2", step2)
sub_app = subgraph.compile()

# Use in main graph
def use_subgraph(state):
    result = sub_app.invoke(state)
    return result

workflow.add_node("subprocess", use_subgraph)
```

---

## Comparison Matrix

| Feature | LangFlow | LangSmith | Langfuse | LangGraph |
|---------|----------|-----------|----------|-----------|
| **Purpose** | Visual workflow builder | Monitoring & testing | Open-source monitoring | Stateful agents |
| **Interface** | GUI (drag-and-drop) | Web dashboard | Web dashboard | Code-first |
| **Self-Hosted** | ✓ | ✗ (Cloud only) | ✓ | ✓ |
| **Pricing** | Free | Paid plans | Free/Paid | Free |
| **Best For** | Rapid prototyping | Production monitoring | Privacy-focused monitoring | Complex agent workflows |
| **Learning Curve** | Low | Medium | Medium | High |
| **LangChain Integration** | Native | Native | Via callback | Native |

## Recommended Stack

### For Beginners
- **LangFlow** for building
- **LangSmith** for monitoring

### For Production
- **LangGraph** for complex agents
- **Langfuse** for self-hosted monitoring
- **LangSmith** for testing/evaluation

### For Privacy-First
- **LangFlow** (self-hosted)
- **Langfuse** (self-hosted)
- **LangGraph** for orchestration

## Resources

### LangFlow
- Docs: https://docs.langflow.org
- GitHub: https://github.com/logspace-ai/langflow

### LangSmith
- Website: https://smith.langchain.com
- Docs: https://docs.smith.langchain.com

### Langfuse
- Website: https://langfuse.com
- Docs: https://langfuse.com/docs
- GitHub: https://github.com/langfuse/langfuse

### LangGraph
- Docs: https://langchain-ai.github.io/langgraph
- GitHub: https://github.com/langchain-ai/langgraph
- Tutorials: https://langchain-ai.github.io/langgraph/tutorials

---

*This guide covers the essential LangChain ecosystem tools. Each tool serves a specific purpose in the LLM application development lifecycle.*
