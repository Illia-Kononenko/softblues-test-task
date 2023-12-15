import textwrap
import time

from dotenv import find_dotenv, load_dotenv
from selenium import webdriver

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()
url = "https://www.amazon.co.uk/gp/help/customer/display.html?nodeId=GKM69DUUYKQWKWX7"


def create_db_from_url(url: str) -> FAISS:
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(3)
    driver.refresh()
    time.sleep(3)
    content = driver.page_source

    with open("page.html", "w", encoding="utf-8") as file:
        file.write(content)

    driver.quit()
    loader = UnstructuredHTMLLoader("page.html")
    transcript = loader.load()
    print(transcript)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about web 
        documents and articles.

        Answer the following question: {question}
        By searching the following information: {docs}

        Only use the factual information from the docs to answer the question.

        If you feel like you don't have enough information to answer the question, 
        say "I don't know, please, contact customer support".

        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


db = create_db_from_url(url)
while True:
    query = input()
    if query != "quit":
        response, docs = get_response_from_query(db, query)
        print(textwrap.fill(response, width=85))
    else:
        break


#query = "What is this article about?"
