import os
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import PyMuPDFLoader

HF_EMBED = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')  # Using a basic sentence embedder model for
# generating the embeddings


def prepare_bot():
    """
    Preparing the vector database on the pdf and saving the db on filesystem
    :return:
    """
    path = os.path.join(os.getcwd(), "processed_data", "Concepts_of_Biology.pdf")
    os.makedirs(os.path.join(os.getcwd(), "vector_store"), exist_ok=True)
    loader = PyMuPDFLoader(path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
    docs = text_splitter.split_documents(data)
    embeddings = HF_EMBED
    Chroma.from_documents(docs, embeddings, persist_directory=os.path.join(os.getcwd(), "vector_store"))


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def user_chat_rag(inp):
    """
    :param inp: User Input
    :return: Response from LLM
    """
    retreiver = Chroma(persist_directory=os.path.join(os.getcwd(), "vector_store"), embedding_function=HF_EMBED)
    retriever = retreiver.as_retriever(k=5)
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", task="text-generation",
                         huggingfacehub_api_token="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")
    template = """
    Answer the following question based on only the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = llm

    # Langchain Expression Language LCEL
    chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )
    return chain.invoke(inp)


if __name__ == '__main__':
    prepare_bot()
    print(user_chat_rag("What is a molecule"))
