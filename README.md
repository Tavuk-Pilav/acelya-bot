# Açelya

This application was developed as part of the Apsiyon Hackathon. It's a RAG-LLM-GenAI based chatbot designed to answer questions on customers' minds. You send your question and the chatbot generates the most appropriate answer from vectordb.

### Features 
- RAG
- make an appointment
- Dictionary
- Convincing



### Technologies Used
- [OpenAI](https://platform.openai.com/docs/api-reference/introduction) - OpenAI version: 1.35.14
- [LangChain](https://python.langchain.com/v0.2/docs/introduction/) - LangChain version: 0.2.5
- [Streamlit](https://docs.streamlit.io/) - Streamlit version: 1.36.0


### Architecture



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