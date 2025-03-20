# City Digital Tour Guide Chat Bot Assistant AI Agent

A comprehensive AI-powered digital tour guide that leverages voice interaction, real-time web search, and advanced language models to provide detailed city information and recommendations. This project integrates several technologies including Streamlit for the user interface, LangChain with Groq API for intelligent responses, DuckDuckGo for search capabilities, and voice interfaces for an interactive experience.

## Overview

The **City Digital Tour Guide Chat Bot Assistant AI Agent** is designed to:
- **Interact via voice and text:** Users can ask questions about cities using voice commands or text input.
- **Search and visualize data:** Integrates with DuckDuckGo for real-time searches and generates data visualizations such as quality metrics, city statistics, and attraction breakdowns.
- **Leverage AI for smart responses:** Utilizes the Groq API with LangChain to process queries and provide accurate, context-aware information about cities.
- **Support customizable settings:** Easily configure city parameters, API keys, and voice settings through environment variables and the Streamlit interface.

## Features

- **Voice Recognition and Synthesis:** 
  - Use the microphone to ask questions.
  - Receive answers via both text and auto-played audio.
- **Real-time Search Integration:**
  - Leverage DuckDuckGo for retrieving up-to-date city information.
- **AI-powered Responses:**
  - Use Groq’s language model (Llama 3 70B) to generate concise and accurate responses.
- **Data Visualizations:**
  - Generate charts and graphs (radar, bar, pie) to visualize city metrics, statistics, and attractions.
- **User-Friendly Interface:**
  - Built with Streamlit for a responsive and interactive user experience.
  - Sidebar for configuring city settings and API keys.
  
## Technologies Used

- **Frontend/UI:** Streamlit, Streamlit Lottie, Streamlit Option Menu
- **Voice Interaction:** SpeechRecognition, gTTS (Google Text-to-Speech)
- **Search Integration:** DuckDuckGo Search
- **AI and Language Processing:** LangChain, Groq API (ChatGroq)
- **Data Visualization:** Matplotlib, Plotly, Seaborn, Pandas, NumPy
- **Utilities:** Requests, Python-dotenv

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/harshadwarokar/city-digital-tour-guide-chat-bot-assistant-ai-agent.git
   cd city-digital-tour-guide-chat-bot-assistant-ai-agent
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirement.txt
   ```

4. **Configure Environment Variables:**

   - Create a `.env` file in the root directory (or update the existing one) with your settings. For example:
     
     ```ini
     GROQ_API_KEY=your_actual_groq_api_key
     DEBUG=True
     DEFAULT_CITY=New York
     LOG_LEVEL=INFO
     DEBUG_MODE=False
     VOICE_RATE=180
     VOICE_VOLUME=0.9
     RECOGNIZER_ENERGY_THRESHOLD=300
     RECOGNIZER_PAUSE_THRESHOLD=0.8
     MAX_SEARCH_RESULTS=5
     ```

## Usage

1. **Run the Application:**

   Start the Streamlit app with the following command:

   ```bash
   streamlit run digitalcityassistant.py
   ```

2. **Set Up the Assistant:**
   - In the sidebar, enter your Groq API key and the city you want to explore.
   - Click on **"Set City"** to initialize the assistant.

3. **Interact with the Assistant:**
   - **Voice Mode:** Use the **"Start Listening"** button to enable voice input. Once finished, click **"Stop Listening"** to process your voice query.
   - **Text Mode:** Type your question in the provided text box and submit it.

4. **View Data Visualizations:**
   - Navigate to the **"City Visualizations"** tab to see various charts and graphs based on sample data (or real-time data in production).

5. **Additional Features:**
   - Test your microphone with the **"Test Microphone"** button.
   - Explore the **"About"** tab for more details on how to use the app and its architecture.

## Configuration

- **Environment Variables:** The application settings are managed via the `.env` file. Ensure you update the `GROQ_API_KEY` and other settings as needed.
- **Voice Settings:** Modify `VOICE_RATE`, `VOICE_VOLUME`, and other speech recognition parameters within the `.env` to adjust voice performance.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please open an issue or contact [Harshad Warokar](https://github.com/harshadwarokar).

---
