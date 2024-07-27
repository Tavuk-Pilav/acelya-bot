import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_PATH = "chroma"
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']  

PROMPT_TEMPLATE = """
Soruyu yalnızca aşağıdaki bağlama dayanarak cevapla:

{context}

Yukarıdaki bağlama dayanarak soruyu cevapla: {question}
"""

def main(query_text):
    # Prepare the DB.
    embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"""Mesajınızı anlayamadım, size yardımcı olabilmemiz için 444 88 88 Telefon Bankacılığımızdan bize ulaşabilir ya da “Müşteri Temsilcisi“ yazarak canlı destek müşteri temsilcimizle görüşebilirsiniz.""")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    response_text = model.invoke(prompt)
    print(response_text.content)

if __name__ == "__main__":
    #query_text = "Apsiyon nedir?"
    #query_text = "Apsiyon Life'tan kimler yararlanabilir?"
    #query_text = "Blue Paket'in temel özellikleri nelerdir?"
    query_text = "Apsiyon'un sunduğu eğitimler nelerdir?"
    main(query_text)
