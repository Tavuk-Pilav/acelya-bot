# Açelya

This application was developed as part of the Apsiyon Hackathon. It's a RAG-LLM-GenAI based chatbot designed to answer questions on customers' minds. You send your question and the chatbot generates the most appropriate answer from vectordb.

### Features 
- **RAG:** Can answer all questions related to Apsiyon
- **Dictionary:** Provides definitions for real estate terms
- **Appointment:** The customer can schedule an appointment for a suitable time and day
- **Persuasive:** Uses a convincing and warm language to persuade the customer
- **Conversation Logs:** The customer representative can review conversation logs and see a summary of the conversation


### Technologies Used
- [OpenAI](https://platform.openai.com/docs/api-reference/introduction) - OpenAI version: 1.35.14
- [LangChain](https://python.langchain.com/v0.2/docs/introduction/) - LangChain version: 0.2.5
- [Streamlit](https://docs.streamlit.io/) - Streamlit version: 1.36.0


### Architecture

![acelyamimari](https://github.com/user-attachments/assets/481cd711-b11c-42d5-bae9-6398c85d20d0)


---

## Requirements

### Environment

Ensure that your Python version is set to `3.10.12` (pip version is `24.1.2`):

```bash
python --version
```
- Setting up Virtualenv:

```bash
pip install virtualenv
```
- Creating a Virtual Environment:
```bash
virtualenv venv
```
- Activating the Virtual Environment:
```bash
source venv/bin/activate
```
- Installing the necessary libraries:
```bash
pip install -r requirements.txt
```

#### Configuration

- Set up your .env file:

```bash
cd <project-directory>
```

```bash
- Create the .env file and add your OPENAI_API_KEY:

    OPENAI_API_KEY='key' # .env file

```
#### Create VectorDB

```bash
python3 create_database.py
```

#### Run

- Launch the Streamlit app in terminal:
```bash
streamlit run app.py
```
