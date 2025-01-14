from model import llm
from classes import GradeDocuments,GraphState
from langchain_core.prompts import ChatPromptTemplate
from rag_store import retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain import hub
from langchain.schema import Document



wikipedia_tool= WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
web_search_tool=DuckDuckGoSearchResults(max_results=3)



structured_llm_grader=llm.with_structured_output(GradeDocuments)
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader=grade_prompt | structured_llm_grader

system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning.
     Just output the rewritten question, do not add any other text. \n\n"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)
question_rewriter=re_write_prompt | llm | StrOutputParser()



prompt = hub.pull("rlm/rag-prompt")
rag_chain=prompt | llm | StrOutputParser()




def retrieve(state: GraphState):
    print('RETRIEVE')
    question=state["question"]
    documents=retriever.get_relevant_documents(question)
    return {"documents": documents,"question": question}

def grade_documents(state: GraphState):

    print('CHECK RELEVANCE')
    question=state["question"]
    documents=state["documents"]
    
    filtered_docs=[]
    web_search="no"
    for d in documents:
        score=retrieval_grader.invoke({"document": d.page_content, "question": question})
        grade=score.binary_score
        if grade=="yes":
            print('DOCUMENT IS RELEVANT')
            filtered_docs.append(d)
        else:
            print('DOCUMENT IS NOT RELEVANT')
            web_search="yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}



def transform_query(state):
    print('TRANSFORM QUERY')
    question=state["question"]
    documents=state["documents"]
    new_question=question_rewriter.invoke({"question": question})
    return {"documents": state["documents"], "question": new_question}

def generate(state):
    print('GENERATE')
    question=state["question"]
    documents=state["documents"]
    generation=rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def web_search(state):
    print('WEB SEARCH')
    question=state["question"]
    documents=state["documents"]
    search_results=wikipedia_tool.run(question)
    web_results=Document(page_content=search_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "yes":
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"