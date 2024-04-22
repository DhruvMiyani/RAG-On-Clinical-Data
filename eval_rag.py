
#RAG evaluation is quite hard for me , I refer some documentation online

eval_dataset = Dataset.from_csv("haa_develAdmittimes.csv")

eval_dataset

!pip install llama-index -qU

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.metrics.critique import harmfulness
from ragas import evaluate

subject_id	hadm_id	timestamp	observations

def create_ragas_dataset(rag_pipeline, eval_dataset):
  rag_dataset = []
  for row in tqdm(eval_dataset):
    answer = rag_pipeline({"query" : row["timestamp"]})
    rag_dataset.append(
        {"subject_id" : row["subject_id"],
         "answer" : answer["hadm_id"],
         "contexts" : [context.page_content for context in answer haa_develAdmittimes['hadm_id']],
         "observations" : [row["observations"]]
         }
    )

haa_develAdmittimes['combined'] = haa_develAdmittimes['hadm_id'].astype(str) + " at " + haa_develAdmittimes['admittime'].astype(str)



  rag_df = haa_develAdmittimes['combined']
  rag_eval_dataset = Dataset.from_pandas(haa_develAdmittimes['combined'])
  return rag_eval_dataset

def evaluate_ragas_dataset(ragas_dataset):
  result = evaluate(
    ragas_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
  )
  return result

"""Lets create our dataset first:"""

from tqdm import tqdm
import pandas as pd

basic_qa_ragas_dataset = create_ragas_dataset(qa_chain, eval_dataset)

"""Save it for later:"""

basic_qa_ragas_dataset.to_csv("basic_qa_ragas_dataset.csv")

"""And finally - evaluate how it did!"""

basic_qa_result = evaluate_ragas_dataset(basic_qa_ragas_dataset)

basic_qa_result

"""### Testing Other Retrievers

Now we can test our how changing our Retriever impacts our RAGAS evaluation!
"""

def create_qa_chain(medical_retriever):
  primary_qa_llm = llm

  created_qa_chain = RetrievalQA.from_chain_type(
      primary_qa_llm,
      medical_retriever=medical_retriever,
      return_source_documents=True
  )

  return created_qa_chain

"""#### Parent Document Retriever

One of the easier ways we can imagine improving a retriever is to embed our documents into small chunks, and then retrieve a significant amount of additional context that "surrounds" the found context.

You can read more about this method [here](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever)!
"""

!pip install chromadb -qU

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=750)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

vectorstore = Chroma(collection_name="split_parents", embedding_function=embeddings_model)

store = InMemoryStore()

parent_document_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

parent_document_retriever.add_documents(base_docs)

"""Let's create, test, and then evaluate our new chain!"""

parent_document_retriever_qa_chain = create_qa_chain(parent_document_retriever)

parent_document_retriever_qa_chain({"query" : "What is RAG?"})["result"]

pdr_qa_ragas_dataset = create_ragas_dataset(parent_document_retriever_qa_chain, eval_dataset)

pdr_qa_ragas_dataset.to_csv("pdr_qa_ragas_dataset.csv")

pdr_qa_result = evaluate_ragas_dataset(pdr_qa_ragas_dataset)

pdr_qa_result

!pip install -q -U rank_bm25

from langchain.retrievers import BM25Retriever, EnsembleRetriever

text_splitter = RecursiveCharacterTextSplitter()
docs = text_splitter.split_documents(base_docs)

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 1

embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5])

ensemble_retriever_qa_chain = create_qa_chain(ensemble_retriever)

ensemble_retriever_qa_chain({"query" : "What subject id here ?"})["result"]

ensemble_qa_ragas_dataset = create_ragas_dataset(ensemble_retriever_qa_chain, eval_dataset)

ensemble_qa_ragas_dataset.to_csv("ensemble_qa_ragas_dataset.csv")

ensemble_qa_result = evaluate_ragas_dataset(ensemble_qa_ragas_dataset)

ensemble_qa_result
