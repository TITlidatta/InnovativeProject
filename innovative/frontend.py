# PyQt5 chatbot UI with LLM + loop + image upload support

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QTextEdit, QVBoxLayout, QLineEdit,
    QLabel, QFileDialog, QHBoxLayout
)
from PyQt5.QtCore import Qt
import sys
import re
from model import process_uploaded_image, analysis
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI 
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional
import requests
import json 
import os
from PyPDF2 import PdfReader
import tempfile
import base64

# ==== Replace these with your actual values ====
API_KEY = ""  # Insert your actual OpenAI API key
llm = ChatOpenAI(model="gpt-4o", openai_api_key=API_KEY)
url = "https://api.openai.com/v1/chat/completions"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

reader = PdfReader('Kidney.pdf')
text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
em = OpenAIEmbeddings(openai_api_key=API_KEY)
vectorstore = Chroma(persist_directory="./vector_store_chroma_kidney", embedding_function=em)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm2 = ChatOpenAI(model="gpt-4", openai_api_key=API_KEY)
qa_chain = RetrievalQA.from_chain_type(llm=llm2, retriever=retriever)

# Global state
need = {"age": None, "Gender": None, "Symptoms": None}
imagex = None  # Placeholder for image path
file_path=None

def match(text):
    global need, imagex
    nums = re.findall(r'\b\d+\b', text)
    gender_keywords = ['male', 'female', 'man', 'woman', 'nonbinary', 'trans']
    gender_found = [g for g in gender_keywords if g in text.lower()]
    
    if nums:
        need['age'] = nums[0]
        return 'a'
    elif gender_found:
        need['Gender'] = gender_found[0]
        return 'g'
    else:
        # Use OpenAI to extract symptoms
        data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "if the sentence has any symptom Extract the symptoms from the sentence or say 'no' if there is no sentence."},
                {"role": "user", "content": text}
            ]
        }
        res = requests.post(url, headers=headers, json=data)
        if res.ok:
            reply = res.json()["choices"][0]["message"]["content"]
            print(reply)
            if reply.lower() == 'no':
                return 'c' if not imagex else 'd'
            else:
                need['Symptoms'] = reply
                return 's'
    return 'c'


class ChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Assistant (PyQt5)")
        self.resize(600, 600)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message...")

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.handle_send)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)

        self.start_button = QPushButton("Start Chat")
        self.start_button.clicked.connect(self.start_session)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("🧠 Chat Interface"))
        layout.addWidget(self.chat_display)
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.start_button)

        self.setLayout(layout)
        self.message_history = []
        self.session_active = False

    def start_session(self):
        self.message_history = [
            SystemMessage(
                """
                You are a medical assistant. Your job is to collect the following information one-by-one:
                1. Age
                2. Gender
                3. Symptoms
                4. Ask for image upload if not yet uploaded.

                Check if the age is matching a age format or gender is matching a format or if the symptom is matching any symptom if not ask the question again to get relevant data.
                Also after getting relevant symptom and once you have asked to upload image if you still need to ask question ask to upload image properly again then on.
                """
            )
        ]
        self.session_active = True
        self.run_loop()

    def run_loop(self):
        if not self.session_active:
            return
        bot_reply = llm.invoke(self.message_history).content
        self.chat_display.append(f"🩺 Assistant: {bot_reply}")

    def handle_send(self):
        global file_path
        global need
        if not self.session_active:
            self.chat_display.append("⚠️ Please start the session first.")
            return

        user_input = self.input_field.text().strip()
        if not user_input:
            return
        self.chat_display.append(f"👤 You: {user_input}")
        self.message_history.append(HumanMessage(content=user_input))

        ch = match(user_input)
        if ch in ['a', 'g', 's', 'c']:
            self.run_loop()
        else:
            self.chat_display.append("🔍 ANALYZING...")
            xtest=process_uploaded_image(file_path)
            labels =analysis(xtest) # top 2 #####GOT 
            for i in range(len(labels)):
                if self.Validator(labels[i],need):
                    break
            if i==2:
                self.chat_display.append('Your situation seems a bit complicated') #### PRINT IN THE SPACE
            else:
                if labels[i]=='normal':
                    self.chat_display.append('You are absolutely normal. No need to worry.')  #### PRINT IN THE SPACE
                else:
                    response = qa_chain.invoke('Tell the preventions or medications or treatments and remedies of  Kidney'+ labels[i])['result']
                    self.chat_display.append('You might be suffering from ' + labels[i]+ '\n\n'+ response)  #### PRINT IN THE SPACE
                    need={
                        'age':None,
                        'Gender':None,
                        'Symptoms':None
                    }
                self.session_active = False

        self.input_field.clear()

    def photo(self,inputfilepath):
        with open(inputfilepath, 'rb') as image_file:
            image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Tell only 'Yes' or 'No' and no other text, whether if the image given is a CT scan of a kidney"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "detail": "high",
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }]}]}
        
        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"}

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

        if response.status_code == 200:
            response_json = response.json()
            datav=response_json["choices"][0]["message"]["content"]
            return datav
        else:
            print('error')
            return 'No'.lower()
        
    def upload_image(self):
        global imagex
        global file_path
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.chat_display.append("🖼️ Image uploaded successfully.")
        if self.photo(file_path) == 'Yes' :
            print('hrer')
            imagex = True
        else:
            imagex=False

    def Validator(self,label,need):
        age = need['age']
        gender = need['Gender'][0]
        symptoms = need['Symptoms'].replace(', ', ' and ')
        sentence = f"A {age}-year-old {gender} is experiencing {symptoms}."
        if label =='normal':
            response='There is no symptoms for any kidney related issues.'
        else:
            response = qa_chain.invoke('Kidney'+ label +'occur at which age and gender ? What are the symptoms? ')['result']
        sss=f"""
        Our patient is experiencing : {sentence}
        Our knowledge says : {response}
        Tell yes or no if the patient majorly aligns to our knowledge hypothesis."""
        data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a validator who validates whether our proposed solution. You only answer the validation as yes or no"},
                    {"role": "user", "content": sss},
                ]
            }
        responsed = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(data))
        result = responsed.json()
        if result['choices'][0]['message']['content']=='yes' or result['choices'][0]['message']['content']=='Yes':
            return True
        else:
            return False



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatApp()
    window.show()
    sys.exit(app.exec_())

