# 🎭 AI Mood Chatbot

An interactive chatbot that responds based on different moods — Angry 😡, Funny 😂, and Sad 😢.
Built using Python, Streamlit, and Mistral AI via LangChain.

---

## 🚀 Features

* 🎭 Multiple AI personalities:

  * 😡 Angry (short-tempered responses)
  * 😂 Funny (humorous replies)
  * 😢 Sad (melancholic tone)
* 💬 Chat interface using Streamlit
* 🖥️ CLI-based chatbot support
* ⚡ Fast responses using Mistral AI
* 🔁 Chat reset and session memory

---

## 🛠️ Tech Stack

* Python
* Streamlit (UI)
* LangChain
* Mistral AI
* dotenv (for environment variables)

---

## 📂 Project Structure

```
GEN-AI/
│── ui_chatbot.py  # Streamlit UI chatbot
│── chatbot.py     # CLI chatbot
│── .env           # API key
│── requirements.txt
│── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/your-username/ai-mood-chatbot.git
cd ai-mood-chatbot
```

### 2. Create virtual environment

```
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Create `.env` file

Add your Mistral API key:

```
MISTRAL_API_KEY=your_api_key_here
```

---

## ▶️ How to Run

### Run Streamlit UI

```
streamlit run ui_chatbot.py
```

### Run CLI chatbot

```
python chatbot.py
```

---

## 🧠 How It Works

* User selects a mood
* System prompt is set accordingly
* Messages are stored in session
* Mistral AI generates responses based on mood + history

---

## 🌟 Future Improvements

* Add more moods (romantic, sarcastic, etc.)
* Voice input/output
* Chat history export
* User authentication
* Rate limiting for API usage

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📌 Author

Your Name
GitHub: https://github.com/nidhiv1303

---
