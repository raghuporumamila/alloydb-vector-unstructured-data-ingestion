Building a Question-and-Answer application with **LangChain**, **LangGraph**, and **FastAPI** using **AlloyDB** allows you to create a production-ready RAG (Retrieval-Augmented Generation) system.

The following guide shows how to search vector data in AlloyDB and pass that context to an LLM within a state-managed workflow. (Note: To ingest the data in Alloy DB please take a look at [ETL Dataflow GCP Doc] (./ETL_Dataflow_GCP_Readme.md)

### 1. Project Setup

You will need the Google Cloud integration for AlloyDB and LangChain components.

```bash
pip install fastapi uvicorn langchain langchain-google-alloydb-pg langchain-google-vertexai langgraph

```

### 2. AlloyDB Vector Store Search

The first step is to establish a connection to AlloyDB and set up the retriever. AlloyDB uses the `AlloyDBEngine` to manage connection pools.

```python
from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore
from langchain_google_vertexai import VertexAIEmbeddings

# Initialize AlloyDB Engine
engine = AlloyDBEngine.from_instance(
    project_id="your-project",
    region="us-central1",
    cluster="my-cluster",
    instance="my-instance",
    database="postgres"
)

# Initialize Embeddings and Vector Store
embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
vector_store = AlloyDBVectorStore.create_sync(
    engine=engine,
    table_name="your_vector_table",
    embedding_service=embeddings
)

# Create a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

```

---

### 3. Building the LangGraph Workflow

LangGraph manages the "state" of your application. You'll define a **Retriever Node** to fetch data and a **Generator Node** to pass that context to the LLM.

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI

# Define the state schema
class AgentState(TypedDict):
    question: str
    context: str
    answer: str

llm = ChatVertexAI(model="gemini-1.5-flash")

# Node 1: Retrieve context from AlloyDB
def retrieve_node(state: AgentState):
    docs = retriever.get_relevant_documents(state["question"])
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

# Node 2: Generate answer using context
def generate_node(state: AgentState):
    prompt = f"""Use the context below to answer the user's question.
    Context: {state['context']}
    Question: {state['question']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}

# Assemble the Graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app_graph = workflow.compile()

```

---

### 4. Exposing via FastAPI

Finally, wrap the graph execution in a FastAPI endpoint to make it accessible to your frontend or other services.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    # Execute the graph with the user's question
    inputs = {"question": request.question}
    result = await app_graph.ainvoke(inputs)
    return {"answer": result["answer"]}

```

### Key Technical Takeaways

* **Context Passing:** The `AgentState` ensures that the `context` string generated in the retrieval step is automatically available to the generation step.
* **AlloyDB Efficiency:** By using `as_retriever()`, LangChain handles the complex SQL vector similarity queries (using the `<=>` operator in AlloyDB) behind the scenes.
* **State Management:** LangGraph allows you to add complex logic later, such as "re-writing" the question if no relevant context is found in AlloyDB.

