🥗 PantryArbitrage: The Zero-Waste Kitchen Concierge

Google AI Agents Intensive — Capstone Project

PantryArbitrage is a multi-agent AI system designed to solve the problem of household food waste. By acting as a kitchen concierge, it audits your fridge, plans meals based on expiry dates, and calculates the financial and environmental value of cooking at home versus ordering takeout.

🏗️ Architecture

The system utilizes a Multi-Agent architecture powered by Gemini 2.0 Flash:

👁️ Agent A (Auditor): Uses Vision capabilities to scan fridge photos or receipts and create a structured inventory JSON.

👨‍🍳 Agent B (Chef): A reasoning agent that plans meals based on expiring ingredients and user preferences. It uses the DuckDuckGo Search Tool to validate recipes.

📊 Agent C (Analyst): A deterministic Python tool that calculates financial savings (Arbitrage) and Carbon Footprint reduction.

📸 Agent D (Scanner): Monitors fridge changes to track waste and consumption habits.

🛠️ Tech Stack

LLM: Google Gemini 2.0 Flash (via google-generativeai)

Frontend: Streamlit

Tools: * duckduckgo-search (Web Search)

pdfplumber (Receipt Parsing)

PIL (Image Processing)

🚀 How to Run Locally

Clone the repository:

git clone (https://github.com/Navaprabhas/capstone-PantryArbitrage)


Install dependencies:

pip install -r requirements.txt


Run the App:

streamlit run app.py


Enter your API Key:
When the app launches, enter your Google Gemini API key in the sidebar.

💡 Features

Multimodal Inputs: Accepts Fridge Photos, PDF Receipts, and Voice Commands.

Strict JSON Mode: Ensures agents communicate in structured data for reliable pipeline execution.

Sustainability Metrics: Real-time calculation of money and CO2 saved.
