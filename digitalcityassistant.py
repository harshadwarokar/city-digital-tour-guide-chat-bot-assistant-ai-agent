import os
import time
import json
import logging
import threading
import tempfile
import requests
import base64
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

import streamlit as st
from streamlit_lottie import st_lottie

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

import speech_recognition as sr
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool

# Use gTTS for text-to-speech
from gtts import gTTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Exception classes
class CityAssistantException(Exception):
    pass

class ApiKeyError(CityAssistantException):
    pass

class VoiceRecognitionError(CityAssistantException):
    pass

class SearchError(CityAssistantException):
    pass

class LLMError(CityAssistantException):
    pass

class DataVisualizationError(CityAssistantException):
    pass

# Abstract base classes
class VoiceInterface(ABC):
    @abstractmethod
    def listen(self) -> str:
        pass

    @abstractmethod
    def speak(self, text: str) -> Optional[bytes]:
        pass

class SearchEngine(ABC):
    @abstractmethod
    def search(self, query: str) -> List[Dict[str, str]]:
        pass

class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, context: List[Dict[str, str]] = None) -> str:
        pass

# Helper: generate an auto-playing audio HTML snippet
def autoplay_audio(audio_bytes: bytes) -> str:
    if not audio_bytes:
        return ""
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    html = f'<audio autoplay hidden="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    return html


# Concrete implementations
class SpeechRecognitionInterface(VoiceInterface):
    def __init__(self, mic_index: Optional[int] = None):
        # Use default microphone (mic_index is None)
        self.mic_index = None
        self.recognizer = sr.Recognizer()
        # Adjust recognizer settings
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

    def listen(self) -> str:
        try:
            mic_names = sr.Microphone.list_microphone_names()
            if not mic_names:
                raise VoiceRecognitionError("No microphone found. Please connect a microphone.")
            # Always use the default microphone (device_index=None)
            with sr.Microphone(device_index=self.mic_index) as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.info("Listening for speech...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            logger.info("Processing speech to text...")
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Recognized: {text}")
            return text
        except sr.WaitTimeoutError:
            raise VoiceRecognitionError("Listening timed out. No speech detected.")
        except sr.UnknownValueError:
            raise VoiceRecognitionError("Could not understand audio")
        except sr.RequestError as e:
            raise VoiceRecognitionError(f"Speech recognition service error: {str(e)}")
        except Exception as e:
            raise VoiceRecognitionError(f"Voice recognition error: {str(e)}")

    def speak(self, text: str) -> Optional[bytes]:
        try:
            logger.info(f"Converting to speech: {text[:50]}...")
            tts = gTTS(text=text, lang='en')
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()  # Return MP3 audio bytes
        except Exception as e:
            logger.error(f"Error in text-to-speech: {str(e)}")
            return None

class DuckDuckGoSearch(SearchEngine):
    def __init__(self):
        self.ddgs = DDGS()
        self.search_tool = DuckDuckGoSearchRun()

    def search(self, query: str) -> List[Dict[str, str]]:
        try:
            results = list(self.ddgs.text(query, max_results=5))
            if not results:
                logger.info("No results from DDGS, using fallback search.")
                results_text = self.search_tool.run(query)
                results = [{"title": "Search Results", "body": results_text, "href": ""}]
            logger.info(f"Search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise SearchError(f"Failed to search for '{query}': {str(e)}")

class GroqLLM(LLMProvider):
    def __init__(self, api_key: str):
        if not api_key:
            raise ApiKeyError("Groq API key is required")
        self.api_key = api_key
        self.model_name = "llama3-70b-8192"
        try:
            self.chat_model = ChatGroq(
                temperature=0.5,
                groq_api_key=self.api_key,
                model_name=self.model_name
            )
            logger.info(f"Initialized Groq LLM with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Groq LLM: {str(e)}")
            raise LLMError(f"Failed to initialize Groq LLM: {str(e)}")

    def generate_response(self, prompt: str, context: List[Dict[str, str]] = None) -> str:
        try:
            system_message = """You are a helpful city information assistant. 
Provide accurate, concise information about the requested city.
Base your answers on the provided search results when available.
If you don't know something, admit it rather than making up information."""
            if context:
                context_text = "\n\nSearch Results:\n"
                for i, item in enumerate(context, 1):
                    title = item.get('title', 'Untitled')
                    body = item.get('body', 'No content')
                    href = item.get('href', 'No link')
                    context_text += f"{i}. {title}\n{body}\n{href}\n\n"
                system_message += context_text
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            logger.info(f"Generating response for prompt: {prompt[:50]}...")
            response = self.chat_model.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM error: {str(e)}")
            raise LLMError(f"Failed to generate response: {str(e)}")

# City Assistant Agent with continuous listening
class CityAssistant:
    def __init__(self, groq_api_key: str, mic_index: Optional[int] = None):
        self.voice_interface = SpeechRecognitionInterface(mic_index=mic_index)
        self.search_engine = DuckDuckGoSearch()
        self.llm_provider = GroqLLM(groq_api_key)
        self.city_name = None
        self.conversation_history = []
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.tools = [
            Tool(
                name="DuckDuckGoSearch",
                func=self.search_engine.search,
                description="Useful for searching for information about cities, attractions, weather, etc."
            )
        ]
        # Continuous listening variables
        self.is_listening = False
        self.last_voice_query = ""
        self.stop_listening_callback = None
        logger.info("City Assistant initialized")

    def set_city(self, city_name: str) -> None:
        self.city_name = city_name
        self.conversation_history.append({
            "role": "system",
            "content": f"The user has selected {city_name} as their city of interest."
        })
        logger.info(f"City set to: {city_name}")

    def process_query(self, query: str) -> str:
        try:
            if not self.city_name:
                return "Please set a city first."
            enriched_query = f"{query} in {self.city_name}" if self.city_name.lower() not in query.lower() else query
            search_results = self.search_engine.search(enriched_query)
            response = self.llm_provider.generate_response(enriched_query, search_results)
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I'm having trouble processing your query. {str(e)}"

    # Background callback accumulates recognized phrases
    def _voice_callback(self, recognizer, audio):
        try:
            query = recognizer.recognize_google(audio)
            logger.info(f"Background recognized: {query}")
            if self.last_voice_query:
                self.last_voice_query += " " + query
            else:
                self.last_voice_query = query
        except Exception as e:
            logger.error(f"Error in voice callback: {str(e)}")

    def start_voice_assistant(self) -> Optional[bytes]:
        if self.is_listening:
            return None
        self.is_listening = True
        self.last_voice_query = ""
        self.stop_listening_callback = self.voice_interface.recognizer.listen_in_background(
            sr.Microphone(device_index=self.voice_interface.mic_index),
            self._voice_callback,
            phrase_time_limit=10
        )
        return self.voice_interface.speak("Listening")

    def stop_voice_assistant(self) -> Tuple[str, str, Optional[bytes], Optional[bytes]]:
        if self.is_listening and self.stop_listening_callback:
            self.stop_listening_callback(wait_for_stop=False)
            self.is_listening = False
            if not self.last_voice_query.strip():
                error_msg = "No voice input detected."
                error_audio = self.voice_interface.speak(error_msg)
                return "", error_msg, None, error_audio
            transcript = f"You said: {self.last_voice_query}"
            transcript_audio = self.voice_interface.speak(transcript)
            response = self.process_query(self.last_voice_query)
            response_audio = self.voice_interface.speak(response)
            return transcript, response, transcript_audio, response_audio
        else:
            return "", "", None, None

# Data visualization components (unchanged)
class CityDataVisualizer:
    def __init__(self):
        pass

    def create_city_comparison_chart(self, city_name: str, metrics: Dict[str, float]) -> plt.Figure:
        try:
            categories = list(metrics.keys())
            values = list(metrics.values())
            plt.close('all')
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid')
            ax.fill(angles, values, alpha=0.25)
            plt.xticks(angles[:-1], categories, size=12)
            ax.set_rlabel_position(0)
            plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=10)
            plt.ylim(0, 10)
            plt.title(f"{city_name} Quality Metrics", size=15, y=1.1)
            return fig
        except Exception as e:
            logger.error(f"Error creating city comparison chart: {str(e)}")
            raise DataVisualizationError(f"Failed to create city comparison chart: {str(e)}")

    def create_city_stats_bar_chart(self, city_name: str, stats: Dict[str, float]) -> plt.Figure:
        try:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 6))
            categories = list(stats.keys())
            values = list(stats.values())
            bars = ax.bar(categories, values, color='skyblue')
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            ax.set_xlabel('Categories')
            ax.set_ylabel('Values')
            ax.set_title(f'{city_name} Statistics')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Error creating city stats bar chart: {str(e)}")
            raise DataVisualizationError(f"Failed to create city stats bar chart: {str(e)}")

    def create_city_attractions_pie_chart(self, city_name: str, attractions: Dict[str, int]) -> plt.Figure:
        try:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 8))
            categories = list(attractions.keys())
            values = list(attractions.values())
            wedges, texts, autotexts = ax.pie(
                values, 
                autopct='%1.1f%%',
                textprops=dict(color="w"),
                shadow=True,
                startangle=90
            )
            ax.legend(
                wedges, 
                categories,
                title="Attraction Types",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1)
            )
            plt.title(f'{city_name} Attractions by Category')
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Error creating city attractions pie chart: {str(e)}")
            raise DataVisualizationError(f"Failed to create city attractions pie chart: {str(e)}")

# Streamlit UI (microphone selection removed, using default mic)
class CityAssistantUI:
    def __init__(self):
        st.set_page_config(
            page_title="City Voice Assistant",
            page_icon="🏙️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        if 'city_assistant' not in st.session_state:
            st.session_state.city_assistant = None
        if 'city_name' not in st.session_state:
            st.session_state.city_name = ""
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []
        if 'visualizer' not in st.session_state:
            st.session_state.visualizer = CityDataVisualizer()
        if 'api_key_input' not in st.session_state:
            st.session_state.api_key_input = ""
        self._set_custom_css()

    def _set_custom_css(self):
        st.markdown("""
        <style>
        .main-title { font-size: 3rem; font-weight: 700; color: #1E88E5; text-align: center; margin-bottom: 1rem; }
        .sub-title { font-size: 1.5rem; font-weight: 500; color: #424242; text-align: center; margin-bottom: 2rem; }
        .user-message { background-color: #E3F2FD; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; }
        .assistant-message { background-color: #F5F5F5; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; }
        .info-box { background-color: #E8F5E9; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 5px solid #4CAF50; }
        .error-box { background-color: #FFEBEE; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 5px solid #F44336; }
        .warning-box { background-color: #FFF8E1; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 5px solid #FFC107; }
        .stButton>button { width: 100%; border-radius: 20px; font-weight: 500; height: 3rem; }
        .city-input { border-radius: 10px; font-size: 1.2rem; }
        </style>
        """, unsafe_allow_html=True)

    def _load_lottie_animation(self, url: str) -> Dict:
        try:
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception as e:
            logger.error(f"Error loading Lottie animation: {str(e)}")
            return None

    def render(self):
        st.markdown('<h1 class="main-title">🏙️ City Voice Assistant</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-title">Explore cities with voice interaction powered by AI</p>', unsafe_allow_html=True)
        
        with st.sidebar:
            st.title("Settings")
            groq_api_key = st.text_input(
                "Enter your Groq API Key",
                type="password",
                value=st.session_state.api_key_input,
                help="Required to use the Groq LLM API"
            )
            st.session_state.api_key_input = groq_api_key

            city_name = st.text_input(
                "Enter City Name",
                value=st.session_state.city_name,
                placeholder="e.g., New York",
                help="The city you want to explore"
            )

            if st.button("Set City", key="set_city"):
                if not groq_api_key:
                    st.error("Please enter your Groq API Key")
                elif not city_name:
                    st.error("Please enter a city name")
                else:
                    try:
                        st.session_state.city_assistant = CityAssistant(groq_api_key, mic_index=None)
                        st.session_state.city_assistant.set_city(city_name)
                        st.session_state.city_name = city_name
                        st.session_state.conversation = []
                        st.success(f"City set to {city_name}")
                    except Exception as e:
                        st.error(f"Error setting city: {str(e)}")
            
            st.subheader("About")
            st.write("""
            This application allows you to explore information about cities using your voice.
            Set a city, then use the "Start Listening" and "Stop Listening" buttons on the Voice Assistant tab.
            You can also test the microphone input using the "Test Microphone" button.
            """)

        tab1, tab2, tab3 = st.tabs(["Voice Assistant", "City Visualizations", "About"])
        
        with tab1:
            if not st.session_state.city_name:
                st.info("Please enter a city name in the sidebar to get started.")
                welcome_animation = self._load_lottie_animation("https://assets5.lottiefiles.com/packages/lf20_KU3FLA.json")
                if welcome_animation:
                    st_lottie(welcome_animation, height=400, key="welcome_animation")
            else:
                st.subheader(f"Exploring {st.session_state.city_name}")
                # Three control buttons: Start, Stop, and Test Microphone.
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Start Listening", key="start_listening"):
                        listen_audio = st.session_state.city_assistant.start_voice_assistant()
                        if listen_audio:
                            st.markdown(autoplay_audio(listen_audio), unsafe_allow_html=True)
                        st.info("Listening...")
                with col2:
                    if st.button("Stop Listening", key="stop_listening"):
                        transcript, response, transcript_audio, response_audio = st.session_state.city_assistant.stop_voice_assistant()
                        if transcript:
                            st.session_state.conversation.append({"role": "user", "content": f"[Voice] {transcript}"})
                        if response:
                            st.session_state.conversation.append({"role": "assistant", "content": f"[Voice] {response}"})
                        html_audio = ""
                        if transcript_audio:
                            html_audio += autoplay_audio(transcript_audio)
                        if response_audio:
                            html_audio += autoplay_audio(response_audio)
                        if html_audio:
                            st.markdown(html_audio, unsafe_allow_html=True)
                with col3:
                    if st.button("Test Microphone", key="test_microphone"):
                        try:
                            recognized_text = st.session_state.city_assistant.voice_interface.listen()
                            test_audio = st.session_state.city_assistant.voice_interface.speak("Test complete. You said: " + recognized_text)
                            st.session_state.conversation.append({"role": "user", "content": f"[Test] {recognized_text}"})
                            st.session_state.conversation.append({"role": "assistant", "content": "[Test] Microphone input working."})
                            st.markdown(autoplay_audio(test_audio), unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error testing microphone: {str(e)}")
                
                with st.form(key="text_question_form"):
                    text_question = st.text_input(
                        f"Type your question about {st.session_state.city_name}",
                        placeholder="e.g., What are the top attractions?",
                        key="text_question"
                    )
                    submit_button = st.form_submit_button("Ask Question")
                if submit_button and text_question:
                    try:
                        if st.session_state.city_assistant:
                            response = st.session_state.city_assistant.process_query(text_question)
                            st.session_state.conversation.append({"role": "user", "content": text_question})
                            st.session_state.conversation.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
                conversation_container = st.container()
                with conversation_container:
                    for message in st.session_state.conversation:
                        if message["role"] == "user":
                            st.markdown(f'<div class="user-message">🧑‍💼 **You**: {message["content"]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="assistant-message">🤖 **Assistant**: {message["content"]}</div>', unsafe_allow_html=True)
        
        with tab2:
            if not st.session_state.city_name:
                st.info("Please enter a city name in the sidebar to view visualizations.")
            else:
                st.subheader(f"Data Visualizations for {st.session_state.city_name}")
                st.markdown(
                    '<div class="warning-box">⚠️ The following visualizations are for demonstration purposes. In production, these would be populated with real data from APIs.</div>',
                    unsafe_allow_html=True
                )
                city_metrics = {
                    "Safety": 8.5,
                    "Affordability": 6.2,
                    "Transportation": 7.8,
                    "Healthcare": 8.1,
                    "Education": 7.9,
                    "Recreation": 8.7,
                    "Environment": 7.0
                }
                city_stats = {
                    "Population (M)": 8.4,
                    "Area (sq km)": 784,
                    "Density (per sq km)": 10712,
                    "GDP (B $)": 1.5,
                    "Unemployment (%)": 5.6
                }
                city_attractions = {
                    "Museums": 35,
                    "Parks": 28,
                    "Restaurants": 45,
                    "Historical Sites": 22,
                    "Shopping": 18,
                    "Entertainment": 12
                }
                viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Quality Metrics", "City Stats", "Attractions"])
                with viz_tab1:
                    st.subheader("City Quality Metrics")
                    try:
                        fig = st.session_state.visualizer.create_city_comparison_chart(
                            st.session_state.city_name,
                            city_metrics
                        )
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error creating visualization: {str(e)}")
                with viz_tab2:
                    st.subheader("City Statistics")
                    try:
                        fig = st.session_state.visualizer.create_city_stats_bar_chart(
                            st.session_state.city_name,
                            city_stats
                        )
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error creating visualization: {str(e)}")
                with viz_tab3:
                    st.subheader("Attractions by Category")
                    try:
                        fig = st.session_state.visualizer.create_city_attractions_pie_chart(
                            st.session_state.city_name,
                            city_attractions
                        )
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error creating visualization: {str(e)}")
                st.subheader("Data Analysis")
                st.write("""
                Based on the visualizations, this city scores highly in recreation and safety metrics,
                while affordability is relatively lower. The city has many cultural attractions, particularly museums and restaurants.
                """)
        
        with tab3:
            st.subheader("About City Voice Assistant")
            st.write("""
            **City Voice Assistant** is an AI-powered application that provides information about cities through voice interaction.
            Technologies used include:
            - **Streamlit** for the UI
            - **LangChain** for AI orchestration
            - **Groq API** for the language model
            - **DuckDuckGo Search** for real-time web search
            - **Speech Recognition & Synthesis** for voice interaction
            """)
            st.subheader("Application Architecture")
            architecture_diagram = """
            graph TD
                User[User] <--> UI[Streamlit UI]
                UI <--> VA[Voice Assistant]
                VA <--> VR[Voice Recognition]
                VA <--> TTS[Text-to-Speech]
                VA <--> CA[City Assistant]
                CA <--> SE[Search Engine]
                CA <--> LLM[Groq LLM]
                SE --> Web[Web Data]
                CA --> Viz[Data Visualization]
                classDef user fill:#f9f,stroke:#333,stroke-width:2px;
                classDef ui fill:#bbf,stroke:#33f,stroke-width:2px;
                classDef voice fill:#fbb,stroke:#f33,stroke-width:2px;
                classDef core fill:#bfb,stroke:#3f3,stroke-width:2px;
                classDef data fill:#ffb,stroke:#ff3,stroke-width:2px;
                class User user;
                class UI,Viz ui;
                class VA,VR,TTS voice;
                class CA,SE,LLM core;
                class Web data;
            """
            st.markdown(f"```mermaid\n{architecture_diagram}\n```")
            st.subheader("How to Use")
            st.write("""
            1. Enter your Groq API key and a city name in the sidebar, then click "Set City".
            2. On the Voice Assistant tab, click "Start Listening" to begin recording (the assistant will say "Listening...").
            3. When finished, click "Stop Listening" – the assistant will immediately repeat your query and answer it via auto-playing audio.
            4. You can also test the microphone input using the "Test Microphone" button.
            5. You can also type questions directly in the text box.
            6. View data visualizations on the City Visualizations tab.
            """)
            st.subheader("Example Questions")
            example_questions = [
                "What are the top tourist attractions?",
                "What's the weather like throughout the year?",
                "Tell me about the local cuisine",
                "What's the public transportation system like?",
                "What are some interesting historical facts?",
                "What neighborhoods should I explore?",
                "What's the cost of living?"
            ]
            for question in example_questions:
                st.markdown(f"- {question}")
            st.subheader("Credits")
            st.write("""
            - Developed as a demonstration of AI agent capabilities
            - Uses Groq API for language model processing
            - Voice recognition powered by Google Speech Recognition API
            - Search capabilities powered by DuckDuckGo
            """)

def main():
    try:
        ui = CityAssistantUI()
        ui.render()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
