from langgraph.graph import START,END,StateGraph
from classes import GraphState
from tools import retrieve,grade_documents,generate,transform_query,web_search,decide_to_generate

workflow=StateGraph(GraphState)

workflow.add_node("retriever",retrieve)
workflow.add_node("grader",grade_documents)
workflow.add_node("generate",generate)
workflow.add_node("transform_query",transform_query)
workflow.add_node("web_search_node",web_search)

workflow.add_edge(START,"retriever")
workflow.add_edge("retriever", "grader")
workflow.add_conditional_edges("grader",decide_to_generate,{
        "transform_query": "transform_query",
        "generate": "generate",
    })
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

app=workflow.compile()
