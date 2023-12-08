# This is a flask app that will run the Financial Chatbot in Flask
from flask import Flask, render_template, request, jsonify
from financialChatbot import FinancialChatbot
import pandas as pd

app = Flask(__name__)

# read .csv of questions and answers
filepath = 'financial_questions_answers.csv'
# questions = data['Question']
# answers = data['Answer']

# chatbot
chatbot = FinancialChatbot(filepath)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['messageText']
    print(user_input)
    response = chatbot.get_response(user_input)
    
    # return response in JSON format
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True)