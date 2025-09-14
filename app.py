# app.py - Real-time Voice AI Patient Simulator (ChatGPT-style)
import streamlit as st
import openai
import json
import datetime
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import pandas as pd
import re
import time
import base64
from io import BytesIO
import asyncio

# Page configuration
st.set_page_config(
    page_title="AI Patient Simulator", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'patient_config' not in st.session_state:
    st.session_state.patient_config = None
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'rapport_level' not in st.session_state:
    st.session_state.rapport_level = 5.0
if 'patient_openness' not in st.session_state:
    st.session_state.patient_openness = 3.0
if 'show_actions' not in st.session_state:
    st.session_state.show_actions = True
if 'voice_mode' not in st.session_state:
    st.session_state.voice_mode = False
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None

# Data structures (same as before)
@dataclass
class CoreTraits:
    emotional_intensity: float = 5.0
    mood_stability: float = 5.0
    anger_reactivity: float = 5.0
    emotional_awareness: float = 5.0
    trust_level: float = 5.0
    attachment_anxiety: float = 5.0
    boundary_awareness: float = 5.0
    social_withdrawal: float = 5.0
    catastrophic_thinking: float = 5.0
    black_white_thinking: float = 5.0
    self_criticism: float = 5.0
    concentration_ability: float = 5.0
    verbal_expressiveness: float = 5.0
    emotional_openness: float = 5.0
    defensiveness: float = 5.0
    response_detail_level: float = 5.0

@dataclass
class DisorderTraits:
    abandonment_sensitivity: float = 0.0
    identity_instability: float = 0.0
    impulsivity: float = 0.0
    self_harm_risk: float = 0.0
    dissociation_frequency: float = 0.0
    worry_intensity: float = 0.0
    physical_anxiety: float = 0.0
    avoidance_behaviors: float = 0.0
    perfectionism: float = 0.0
    control_need: float = 0.0
    hopelessness: float = 0.0
    energy_level: float = 10.0
    anhedonia: float = 0.0
    guilt_shame: float = 0.0
    suicidal_ideation: float = 0.0

@dataclass
class PatientConfig:
    name: str
    age: int
    gender: str
    diagnosis: str
    background_story: str
    core_traits: CoreTraits
    disorder_traits: DisorderTraits
    session_context: str = ""

# Patient templates
PATIENT_TEMPLATES = {
    "emma_bpd": PatientConfig(
        name="Emma",
        age=19,
        gender="Female",
        diagnosis="Borderline Personality Disorder",
        background_story="College student, recent painful breakup, history of unstable relationships, struggles with self-image",
        core_traits=CoreTraits(
            emotional_intensity=9.0, mood_stability=2.0, anger_reactivity=8.0,
            trust_level=3.0, attachment_anxiety=9.0, boundary_awareness=2.0,
            black_white_thinking=8.0, self_criticism=9.0,
            emotional_openness=7.0, defensiveness=8.0
        ),
        disorder_traits=DisorderTraits(
            abandonment_sensitivity=9.0, identity_instability=8.0, 
            impulsivity=7.0, self_harm_risk=6.0, dissociation_frequency=5.0
        ),
        session_context="Emma comes in distressed after her boyfriend broke up with her yesterday."
    ),
    
    "david_mdd": PatientConfig(
        name="David",
        age=45,
        gender="Male",
        diagnosis="Major Depressive Disorder",
        background_story="Recently unemployed executive, financial stress, feels like a failure, withdrawn from family",
        core_traits=CoreTraits(
            mood_stability=2.0, emotional_awareness=3.0,
            trust_level=4.0, social_withdrawal=8.0,
            catastrophic_thinking=8.0, self_criticism=9.0, concentration_ability=3.0,
            verbal_expressiveness=3.0, emotional_openness=2.0, response_detail_level=2.0
        ),
        disorder_traits=DisorderTraits(
            hopelessness=8.0, energy_level=2.0, anhedonia=8.0, 
            guilt_shame=9.0, suicidal_ideation=4.0
        ),
        session_context="David lost his job 3 months ago. He speaks slowly, avoids eye contact, and gives minimal responses."
    ),
    
    "sarah_gad": PatientConfig(
        name="Sarah",
        age=28,
        gender="Female", 
        diagnosis="Generalized Anxiety Disorder",
        background_story="New mother, perfectionist tendencies, overwhelmed by responsibilities, constant worrying",
        core_traits=CoreTraits(
            emotional_intensity=7.0, mood_stability=4.0,
            trust_level=6.0, attachment_anxiety=7.0,
            catastrophic_thinking=9.0, concentration_ability=3.0,
            verbal_expressiveness=8.0, emotional_openness=6.0, defensiveness=5.0
        ),
        disorder_traits=DisorderTraits(
            worry_intensity=9.0, physical_anxiety=8.0, avoidance_behaviors=6.0,
            perfectionism=9.0, control_need=8.0
        ),
        session_context="Sarah is a new mother who can't stop worrying about everything that could go wrong."
    )
}

# OpenAI client
@st.cache_resource
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("⚠️ OpenAI API key not configured. Please add it to Streamlit secrets.")
        st.stop()
    return openai.OpenAI(api_key=api_key)

class RealTimeVoiceManager:
    """Handles real-time voice interaction like ChatGPT"""
    
    def __init__(self):
        self.client = get_openai_client()
    
    def text_to_speech(self, text: str, voice: str = "alloy") -> bytes:
        """Convert text to speech using OpenAI's TTS"""
        try:
            # Remove action descriptions for speech
            clean_text = re.sub(r'\*[^*]*\*', '', text).strip()
            
            response = self.client.audio.speech.create(
                model="tts-1-hd",  # Higher quality for better experience
                voice=voice,
                input=clean_text,
                speed=1.0
            )
            return response.content
        except Exception as e:
            st.error(f"Text-to-speech error: {str(e)}")
            return b""
    
    def speech_to_text(self, audio_bytes: bytes) -> str:
        """Convert speech to text using OpenAI's Whisper"""
        try:
            audio_file = BytesIO(audio_bytes)
            audio_file.name = "audio.webm"
            
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            return transcript
        except Exception as e:
            st.error(f"Speech-to-text error: {str(e)}")
            return ""

class VoicePatientSimulator:
    def __init__(self):
        self.client = get_openai_client()
        self.voice_manager = RealTimeVoiceManager()
    
    def generate_patient_response(self, config: PatientConfig, conversation_history: List[str], 
                                rapport: float, openness: float, include_voice: bool = False) -> tuple:
        
        system_prompt = self.build_system_prompt(config, rapport, openness)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        for i, msg in enumerate(conversation_history[-6:]):
            role = "assistant" if i % 2 == 0 else "user"
            messages.append({"role": role, "content": msg})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            text_response = response.choices[0].message.content
            
            # Generate voice if requested
            audio_bytes = b""
            if include_voice:
                voice = self.select_voice_for_patient(config)
                audio_bytes = self.voice_manager.text_to_speech(text_response, voice)
            
            return text_response, audio_bytes
            
        except Exception as e:
            return f"I'm having trouble responding right now.", b""
    
    def select_voice_for_patient(self, config: PatientConfig) -> str:
        """Select appropriate voice based on patient characteristics"""
        if config.gender.lower() == "male":
            if config.age < 30:
                return "echo"  # Younger male
            else:
                return "onyx"  # Mature male
        else:
            if config.age < 25:
                return "nova"   # Younger female
            elif "Anxiety" in config.diagnosis:
                return "shimmer"  # Softer, anxious
            elif "Depression" in config.diagnosis:
                return "alloy"   # Neutral, subdued
            else:
                return "nova"    # Default female
    
    def build_system_prompt(self, config: PatientConfig, rapport: float, openness: float) -> str:
        return f"""You are {config.name}, a {config.age}-year-old {config.gender.lower()} patient in therapy.

DIAGNOSIS: {config.diagnosis}
BACKGROUND: {config.background_story}
SESSION CONTEXT: {config.session_context}

Current rapport: {rapport:.1f}/10, Openness: {openness:.1f}/10

IMPORTANT: You are having a real conversation. Respond naturally and conversationally.
Keep responses 2-3 sentences maximum. Show your personality and symptoms through your words and tone.
React authentically to what the therapist says."""

class TherapeuticAnalyzer:
    THERAPEUTIC_TECHNIQUES = {
        "validation": ["understand", "makes sense", "hear you", "valid", "difficult"],
        "empathy": ["feel", "sounds", "imagine", "must be", "experiencing"],
        "clarification": ["what do you mean", "tell me more", "help me understand"],
        "reflection": ["you're saying", "sounds like", "it seems", "you feel"],
        "rapport": ["thank you", "appreciate", "brave", "strength", "trust"],
        "cbt": ["thought", "thinking", "evidence", "alternative", "realistic"],
        "acceptance": ["okay", "alright", "understandable", "human", "normal"]
    }
    
    @classmethod
    def analyze_response(cls, therapist_message: str) -> Dict[str, float]:
        message_lower = therapist_message.lower()
        technique_scores = {}
        
        for technique, keywords in cls.THERAPEUTIC_TECHNIQUES.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            technique_scores[technique] = min(score / len(keywords), 1.0)
        
        return technique_scores
    
    @classmethod
    def calculate_rapport_change(cls, technique_scores: Dict[str, float], patient_traits: CoreTraits) -> float:
        positive_impact = (
            technique_scores.get("validation", 0) * 0.3 +
            technique_scores.get("empathy", 0) * 0.3 +
            technique_scores.get("acceptance", 0) * 0.2
        )
        
        rapport_change = positive_impact * 2 * (patient_traits.trust_level / 10)
        return max(-1.0, min(1.0, rapport_change))

@st.cache_resource
def get_patient_simulator():
    return VoicePatientSimulator()

@st.cache_resource  
def get_analyzer():
    return TherapeuticAnalyzer()

# JavaScript for real-time voice recording
def get_voice_recorder_html():
    return """
    <div id="voice-recorder">
        <style>
            .voice-container {
                display: flex;
                align-items: center;
                gap: 15px;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                margin: 10px 0;
            }
            .record-btn {
                width: 80px;
                height: 80px;
                border-radius: 50%;
                border: none;
                background: #ff4b4b;
                color: white;
                font-size: 24px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
            }
            .record-btn:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
            }
            .record-btn.recording {
                background: #00d4aa;
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(0, 212, 170, 0.7); }
                70% { box-shadow: 0 0 0 20px rgba(0, 212, 170, 0); }
                100% { box-shadow: 0 0 0 0 rgba(0, 212, 170, 0); }
            }
            .voice-status {
                color: white;
                font-size: 18px;
                font-weight: 500;
            }
            .voice-instructions {
                color: rgba(255, 255, 255, 0.8);
                font-size: 14px;
            }
        </style>
        
        <div class="voice-container">
            <button class="record-btn" id="recordBtn" onclick="toggleRecording()">
                🎙️
            </button>
            <div>
                <div class="voice-status" id="voiceStatus">Click to start recording</div>
                <div class="voice-instructions">Speak naturally as you would in a therapy session</div>
            </div>
        </div>
        
        <script>
            let mediaRecorder;
            let audioChunks = [];
            let isRecording = false;
            
            async function toggleRecording() {
                const btn = document.getElementById('recordBtn');
                const status = document.getElementById('voiceStatus');
                
                if (!isRecording) {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        mediaRecorder = new MediaRecorder(stream);
                        audioChunks = [];
                        
                        mediaRecorder.ondataavailable = (event) => {
                            audioChunks.push(event.data);
                        };
                        
                        mediaRecorder.onstop = async () => {
                            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                            const reader = new FileReader();
                            reader.onload = function() {
                                const base64Audio = reader.result.split(',')[1];
                                // Send to Streamlit
                                window.parent.postMessage({
                                    type: 'audio_recorded',
                                    audio: base64Audio
                                }, '*');
                            };
                            reader.readAsDataURL(audioBlob);
                            
                            // Stop all tracks
                            stream.getTracks().forEach(track => track.stop());
                        };
                        
                        mediaRecorder.start();
                        isRecording = true;
                        btn.classList.add('recording');
                        btn.innerHTML = '⏹️';
                        status.textContent = 'Recording... Click to stop';
                        
                    } catch (err) {
                        status.textContent = 'Microphone access denied. Please enable microphone.';
                        console.error('Error accessing microphone:', err);
                    }
                } else {
                    mediaRecorder.stop();
                    isRecording = false;
                    btn.classList.remove('recording');
                    btn.innerHTML = '🎙️';
                    status.textContent = 'Processing audio...';
                }
            }
            
            // Listen for messages from Streamlit
            window.addEventListener('message', (event) => {
                if (event.data.type === 'transcription_complete') {
                    document.getElementById('voiceStatus').textContent = 'Click to start recording';
                }
            });
        </script>
    </div>
    """

def create_audio_player(audio_bytes: bytes, auto_play: bool = True) -> str:
    """Create an auto-playing audio player"""
    if not audio_bytes:
        return ""
    
    audio_b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <div style="margin: 10px 0;">
        <audio controls {'autoplay' if auto_play else ''} style="width: 100%;">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
    </div>
    """
    return audio_html

def main():
    st.title("🧠🎙️ AI Patient Simulator - Voice Mode")
    st.markdown("**Real-time voice conversation with AI patients - just like ChatGPT Voice!**")
    
    with st.expander("ℹ️ Voice Instructions", expanded=False):
        st.markdown("""
        **How to use Voice Mode:**
        
        1. **Enable microphone** when prompted by your browser
        2. **Click the red microphone button** to start recording
        3. **Speak naturally** as you would in a therapy session
        4. **Click the stop button** when you're done speaking
        5. **Wait for the patient's response** - it will play automatically
        
        **Tips for best experience:**
        - Speak clearly and at normal volume
        - Use a quiet environment if possible
        - Wait for the patient to finish speaking before responding
        - You can also type responses if preferred
        """)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Voice mode toggle
        st.session_state.voice_mode = st.checkbox(
            "🎙️ Enable Voice Mode", 
            value=st.session_state.voice_mode,
            help="Real-time voice conversation like ChatGPT"
        )
        
        st.divider()
        
        setup_mode = st.radio(
            "Setup Method:",
            ["Pre-built Templates", "Custom Configuration"]
        )
        
        if setup_mode == "Pre-built Templates":
            render_template_selection()
        
        st.divider()
        render_session_controls()
    
    if st.session_state.session_active and st.session_state.patient_config:
        render_voice_chat_interface()
    else:
        render_welcome_screen()

def render_voice_chat_interface():
    """Real-time voice chat interface"""
    
    # Patient info header
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**Patient:** {st.session_state.patient_config.name} ({st.session_state.patient_config.age}, {st.session_state.patient_config.diagnosis})")
        with col2:
            st.metric("Rapport", f"{st.session_state.rapport_level:.1f}/10")
        with col3:
            st.metric("Openness", f"{st.session_state.patient_openness:.1f}/10")
    
    st.divider()
    
    # Chat messages
    for i, (role, message) in enumerate(st.session_state.messages):
        with st.chat_message(role):
            processed_message = process_actions(message, st.session_state.show_actions)
            
            if role == "patient":
                st.markdown(f"**{st.session_state.patient_config.name}**: {processed_message}")
                
                # Auto-play patient audio
                if st.session_state.voice_mode and f"patient_audio_{i}" in st.session_state:
                    audio_html = create_audio_player(st.session_state[f"patient_audio_{i}"], auto_play=True)
                    st.markdown(audio_html, unsafe_allow_html=True)
            else:
                st.markdown(f"**Therapist**: {processed_message}")
    
    # Voice input interface
    if st.session_state.voice_mode:
        st.markdown("### 🎙️ Voice Input")
        
        # Render voice recorder
        st.markdown(get_voice_recorder_html(), unsafe_allow_html=True)
        
        # Handle audio from JavaScript
        if st.session_state.recorded_audio:
            audio_data = base64.b64decode(st.session_state.recorded_audio)
            voice_manager = RealTimeVoiceManager()
            transcribed_text = voice_manager.speech_to_text(audio_data)
            
            if transcribed_text:
                st.success(f"You said: {transcribed_text}")
                handle_therapist_response(transcribed_text)
                st.session_state.recorded_audio = None  # Clear after processing
        
        # Alternative text input
        st.markdown("**Or type your response:**")
        if prompt := st.chat_input("Type your response as the therapist..."):
            handle_therapist_response(prompt)
    else:
        # Standard text-only mode
        if prompt := st.chat_input("Type your response as the therapist..."):
            handle_therapist_response(prompt)

def handle_therapist_response(message: str):
    """Handle therapist response with voice generation"""
    
    st.session_state.messages.append(("therapist", message))
    
    # Analyze therapeutic techniques
    analyzer = get_analyzer()
    techniques = analyzer.analyze_response(message)
    rapport_change = analyzer.calculate_rapport_change(techniques, st.session_state.patient_config.core_traits)
    
    # Update rapport and openness
    st.session_state.rapport_level = max(0, min(10, st.session_state.rapport_level + rapport_change))
    st.session_state.patient_openness = max(0, min(10, st.session_state.patient_openness + rapport_change * 0.5))
    
    # Generate patient response with voice
    simulator = get_patient_simulator()
    patient_response, audio_bytes = simulator.generate_patient_response(
        st.session_state.patient_config, 
        [msg[1] for msg in st.session_state.messages],
        st.session_state.rapport_level, 
        st.session_state.patient_openness,
        include_voice=st.session_state.voice_mode
    )
    
    st.session_state.messages.append(("patient", patient_response))
    
    # Store audio for this message
    if audio_bytes:
        message_index = len(st.session_state.messages) - 1
        st.session_state[f"patient_audio_{message_index}"] = audio_bytes
    
    # Show detected techniques
    detected_techniques = [tech for tech, score in techniques.items() if score > 0]
    if detected_techniques:
        st.info(f"🔍 Detected techniques: {', '.join(detected_techniques)}")
    
    st.rerun()

def process_actions(text, show_actions):
    if show_actions:
        return re.sub(r'\*(.*?)\*', r'***\1***', text)
    else:
        return re.sub(r'\*[^*]*\*', '', text).strip()

def render_template_selection():
    st.subheader("📋 Pre-built Patients")
    
    template_descriptions = {
        "emma_bpd": "**Emma (19, BPD)** 🎭 Young female voice, emotional intensity",
        "david_mdd": "**David (45, Depression)** 🎭 Mature male voice, low energy", 
        "sarah_gad": "**Sarah (28, Anxiety)** 🎭 Anxious female voice, rapid speech"
    }
    
    selected_template = st.radio(
        "Choose a patient:",
        options=list(template_descriptions.keys()),
        format_func=lambda x: template_descriptions[x]
    )
    
    if st.button("Load Patient", type="primary"):
        st.session_state.patient_config = PATIENT_TEMPLATES[selected_template]
        st.session_state.messages = []
        st.session_state.rapport_level = 5.0
        st.session_state.patient_openness = 3.0
        st.session_state.session_active = False
        st.success(f"Loaded {st.session_state.patient_config.name} with voice!")
        st.rerun()

def render_session_controls():
    st.subheader("🎛️ Session Controls")
    
    st.session_state.show_actions = st.checkbox(
        "Show action descriptions", 
        value=st.session_state.show_actions
    )
    
    if not st.session_state.patient_config:
        st.warning("⚠️ Configure a patient first")
        return
    
    if not st.session_state.session_active:
        if st.button("▶️ Start Voice Session", type="primary", use_container_width=True):
            start_session()
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏹️ End Session", use_container_width=True):
                st.session_state.session_active = False
                st.success("Session ended")
                st.rerun()
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.messages = []
                st.session_state.rapport_level = 5.0
                st.session_state.patient_openness = 3.0
                st.rerun()

def start_session():
    st.session_state.session_active = True
    st.session_state.messages = []
    st.session_state.rapport_level = 5.0
    st.session_state.patient_openness = 3.0
    
    simulator = get_patient_simulator()
    initial_response, audio_bytes = simulator.generate_patient_response(
        st.session_state.patient_config, [], 
        st.session_state.rapport_level, 
        st.session_state.patient_openness,
        include_voice=st.session_state.voice_mode
    )
    
    st.session_state.messages.append(("patient", initial_response))
    
    # Store initial audio
    if audio_bytes:
        st.session_state["patient_audio_0"] = audio_bytes
    
    st.rerun()

def render_welcome_screen():
    st.markdown("""
    ## Welcome to Real-Time Voice AI Patient Simulator! 🎙️
    
    Experience therapy practice with **natural voice conversation** - just like ChatGPT Voice mode!
    
    ### 🎯 Voice Features:
    - **🎙️ Click to talk** - No file uploads needed
    - **🔊 Instant patient responses** - Hear them speak immediately  
    - **🎭 Realistic voices** - Different voices for each patient type
    - **⚡ Real-time conversation** - Natural therapy session flow
    
    ### 🚀 Getting Started:
    1. **Enable Voice Mode** in the sidebar
    2. **Select a patient** template (Emma, David, or Sarah)
    3. **Start session** and begin voice conversation
    4. **Click microphone** → **Speak** → **Listen to patient**
    
    **Ready for realistic voice therapy practice?** Configure a patient and start your session!
    """)

if __name__ == "__main__":
    main()
