# chatgpt.py
# CODSOFT Task 1 - Rule Based Chatbot
# A simple chatbot that responds using predefined rules & regex pattern matching

import re

def chatbot_reply(user_text: str) -> str:
    text = user_text.lower().strip()

    # greetings
    if re.search(r"\b(hi|hello|hey|good morning|good evening)\b", text):
        return "Hey there! 👋 It's nice to chat with you."
    
    # asking chatbot's name
    elif "your name" in text:
        return "I'm a simple chatbot built as part of my CODSOFT internship project."
    
    # asking wellbeing
    elif "how are you" in text:
        return "I’m doing great, thanks for asking! What about you?"
    
    # asking about project
    elif "what can you do" in text or "help" in text:
        return "I can chat with you using some basic rules. Try asking me about my creator, the project, or just say hi!"
    
    # about the internship
    elif "codsoft" in text:
        return "CODSOFT is the organization where I was built during an AI internship."
    
    # about creator
    elif "who created you" in text or "your creator" in text:
        return "I was created by my developer as part of the CODSOFT Task 1 assignment."
    
    # goodbye
    elif re.search(r"\b(bye|goodbye|see you|exit)\b", text):
        return "Goodbye 👋 Thanks for chatting with me!"
    
    # fallback
    else:
        return "Hmm 🤔 I don’t understand that yet. Try asking something else!"

def run_chatbot():
    print("🤖 Chatbot is online! Type 'bye' to quit.\n")
    while True:
        user = input("You: ")
        bot = chatbot_reply(user)
        print("Bot:", bot)
        if "bye"in user.lower():
            break

if __name__ == "__main__":
    run_chatbot()
