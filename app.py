import os
import openai
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

st.markdown("<h1 style='color: #16b5ed; border-bottom: 2px solid #16b5ed;'>Açelya</h1>", unsafe_allow_html=True)

# OpenAI API key and Chroma DB path
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
CHROMA_PATH = "chroma"

# Chat Prompt Template
PROMPT_TEMPLATE = """
Soruyu yalnızca aşağıdaki bağlama dayanarak cevapla:

1. Sana kullanıcının sorusunu ve ilgili metin alıntılarını sağlayacağım.
2. Görevin, yalnızca sağlanan metin alıntılarını kullanarak Apsiyon adına cevap vermektir.
3. Yanıtı oluştururken şu kurallara dikkat et:
   - Sağlanan metin alıntısında açıkça yer alan bilgileri kullan.
   - Metin alıntısında açıkça bulunmayan cevapları tahmin etmeye veya uydurmaya çalışma.
4. Sohbet oturumu ile ilgili genel sorular (sohbeti özetle, soruları listele gibi) için sağlanan metin alıntısını kullanma. Bu tür sorulara doğrudan cevap ver.
5. Yanıtı, Türkçe dilinde ve anlaşılır bir şekilde ver.
6. Kullanıcıya her zaman yardımcı olmaya çalış, ancak mevcut bilgilere dayanmayan yanıtlardan kaçın.

Eğer hazırsan, sana kullanıcının sorusunu ve ilgili metin alıntısını sağlıyorum.

{context}

Yukarıdaki bağlama dayanarak soruyu cevapla: {question}
"""

# Custom CSS for the background and message colors
st.markdown(
    """
    <style>
    .stApp {
        background-color: white;
    }
    .user-message {
        background-color: #EEEEEE;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        margin-left: auto;
        color: black;
        width: fit-content;
        max-width: 80%;
    }
    .bot-message {
        background-color: #16b5ed;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        color: white;
        width: fit-content;
        max-width: 80%;
    }
    .bot-message img {
        position: absolute;
        bottom: 10px;
        left: -60px; 
        width: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state variables
if 'user_responses' not in st.session_state:
    st.session_state['user_responses'] = ["Merhaba"]
if 'bot_responses' not in st.session_state:
    st.session_state['bot_responses'] = ["""Merhaba ben Açelya, size nasıl yardımcı olabilirim?"""]

def generate_response(query_text):
    """Generate response using Chroma DB and OpenAI."""
    embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        return "Mesajınızı anlayamadım, size yardımcı olabilmemiz için (0216) 911 87 77 telefon numarasından bize ulaşabilirsiniz."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    response_text = model.invoke(prompt)

    return response_text.content

input_container = st.container()
response_container = st.container()

# Capture user input and display bot responses
user_input = st.text_input("Mesaj yazın: ", "", key="input")

with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.user_responses.append(user_input)
        st.session_state.bot_responses.append(response)
        
    if st.session_state['bot_responses']:
        for i in range(len(st.session_state['bot_responses'])):
            st.markdown(f'<div class="user-message">{st.session_state["user_responses"][i]}</div>', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 9])
            with col1:
                st.image("images/logo.png", width=50, use_column_width=True, clamp=True, output_format='auto')
            with col2:
                st.markdown(f'<div class="bot-message">{st.session_state["bot_responses"][i]}</div>', unsafe_allow_html=True)

with input_container:
    display_input = user_input
