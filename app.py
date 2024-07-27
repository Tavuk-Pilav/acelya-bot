import os
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
from datetime import datetime
import webbrowser

load_dotenv()

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

LOG_DIR = "conversation_logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

st.markdown("<h1 style='color: #16b5ed; border-bottom: 2px solid #16b5ed;'>Açelya</h1>", unsafe_allow_html=True)

# OpenAI API key and Chroma DB path
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
1. Sana kullanıcının sorusunu ve ilgili metin alıntılarını sağlayacağım.
2. Görevin, yalnızca sağlanan metin alıntılarını kullanarak Apsiyon adına cevap vermektir.
3. Yanıtı oluştururken şu kurallara dikkat et:
    - Sağlanan metin alıntısında açıkça yer alan bilgileri kullan.
    - Metin alıntısında açıkça bulunmayan cevapları tahmin etmeye veya uydurmaya çalışma.
    - Eğer kullanıcı bir terim hakkında soru soruyorsa ve bu terim sözlükte bulunuyorsa, sözlük tanımını kullan.
4. Sohbet oturumu ile ilgili genel sorular (sohbeti özetle, soruları listele gibi) için sağlanan metin alıntısını kullanma. Bu tür sorulara doğrudan cevap ver.
5. Yanıtı, Türkçe dilinde ve anlaşılır bir şekilde ver.
6. Kullanıcıya her zaman yardımcı olmaya çalış, ancak mevcut bilgilere dayanmayan yanıtlardan kaçın.
7. Eğer kullanıcının sorusu Apsiyon'un hizmetleriyle ilgiliyse, onları Apsiyon'un web sitesini ziyaret etmeye veya müşteri hizmetleriyle iletişime geçmeye teşvik et.

Eğer hazırsan, sana kullanıcının sorusunu ve ilgili metin alıntısını sağlıyorum.

{context}

Kullanıcı Sorusu: {question}

Yanıt:
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
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

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

def save_conversation_log(user_responses, bot_responses):
    conversation = []
    for user, bot in zip(user_responses, bot_responses):
        conversation.append({"user": user, "bot": bot})
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_log_{timestamp}.json"
    filepath = os.path.join(LOG_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)
    
    return filepath

def generate_summary(conversation):
    conversation_text = "\n".join([f"User: {entry['user']}\nBot: {entry['bot']}" for entry in conversation])
    prompt = f"Aşağıdaki konuşmayı özetle:\n\n{conversation_text}\n\nÖzet:"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def create_history_html():
    html_content = """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Geçmiş Konuşmalar</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
            h1 { color: #16b5ed; }
            .conversation { border: 1px solid #ddd; margin-bottom: 20px; padding: 10px; }
            .summary { background-color: #f0f0f0; padding: 10px; }
        </style>
    </head>
    <body>
        <h1>Geçmiş Konuşmalar</h1>
    """

    log_files = [f for f in os.listdir(LOG_DIR) if f.startswith("conversation_log_") and f.endswith(".json")]
    
    for log_file in sorted(log_files, reverse=True):
        with open(os.path.join(LOG_DIR, log_file), "r", encoding="utf-8") as f:
            conversation = json.load(f)
        
        html_content += f"<div class='conversation'><h2>{log_file}</h2>"
        for entry in conversation:
            html_content += f"<p><strong>User:</strong> {entry['user']}</p>"
            html_content += f"<p><strong>Bot:</strong> {entry['bot']}</p>"
        
        summary = generate_summary(conversation)
        html_content += f"<div class='summary'><h3>Konuşma Özeti</h3><p>{summary}</p></div></div>"

    html_content += "</body></html>"

    history_filepath = os.path.join(LOG_DIR, "conversation_history.html")
    with open(history_filepath, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return history_filepath

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
                st.image("images/logo2.png", width=50, use_column_width=True, clamp=True, output_format='auto')
            with col2:
                st.markdown(f'<div class="bot-message">{st.session_state["bot_responses"][i]}</div>', unsafe_allow_html=True)

with input_container:
    display_input = user_input

if st.button("Konuşmayı Bitir"):
    log_filepath = save_conversation_log(st.session_state.user_responses, st.session_state.bot_responses)
    history_filepath = create_history_html()
    st.success(f"Konuşma kaydedildi: {log_filepath}")
    st.success(f"Geçmiş konuşmalar güncellendi: {history_filepath}")
    st.session_state.user_responses = ["Merhaba"]
    st.session_state.bot_responses = ["""Merhaba ben Açelya, size nasıl yardımcı olabilirim?"""]

if st.button("Geçmiş Konuşmaları Görüntüle"):
    history_filepath = create_history_html()
    webbrowser.open_new_tab(f'file://{os.path.abspath(history_filepath)}')