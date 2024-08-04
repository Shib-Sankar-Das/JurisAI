from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)

# Initialize embeddings and load the vector database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("ipc_vector_db", embeddings)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Set up the prompt template
prompt_template = """<s>[INST]This is a chat template and As a legal chat bot specializing in Indian Penal Code queries, your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# Set up the language model
TOGETHER_AI_API = os.environ.get('TOGETHER_AI', '2a7c5dcdbb1049a39117ac0865c4d04008d49db31aa85a3258603817af16dbd0')
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=TOGETHER_AI_API
)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    input_prompt = data.get('message')
    chat_history = data.get('chat_history', [])

    memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)
    for message in chat_history:
        memory.save_context({"input": message['user']}, {"output": message['assistant']})

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

    result = qa({"question": input_prompt})
    
    return jsonify({
        "answer": result["answer"],
        "chat_history": chat_history + [{"user": input_prompt, "assistant": result["answer"]}]
    })

@app.route('/reset', methods=['POST'])
def reset_conversation():
    return jsonify({"message": "Conversation reset successfully"})

if __name__ == '__main__':
    app.run(debug=True)