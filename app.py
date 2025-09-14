# app.py - Complete Fixed AI Patient Simulator
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
import streamlit.components.v1 as components

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
if 'voice_input' not in st.session_state:
    st.session_state.voice_input = ""

# Data structures
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
        session_context="Emma comes in distressed after her boyfriend broke up with her yesterday. She's oscillating between anger and despair."
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
        session_context="Sarah is a new mother who can't stop worrying about everything that could go wrong. She speaks rapidly and seeks constant reassurance."
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

class AIPatientSimulator:
    """AI Patient that responds to therapist"""
    
    def __init__(self):
        self.client = get_openai_client()
    
    def generate_patient_response(self, config: PatientConfig, conversation_history: List[str], 
                                rapport: float, openness: float, include_voice: bool = False) -> tuple:
        """Generate what the PATIENT says in response to the THERAPIST"""
        
        system_prompt = self.build_patient_prompt(config, rapport, openness)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Build conversation: therapist = user, patient = assistant
        for i, msg in enumerate(conversation_history[-6:]):
            if i % 2 == 0:  # Even indices are therapist messages
                messages.append({"role": "user", "content": f"Therapist: {msg}"})
            else:  # Odd indices are patient responses
                messages.append({"role": "assistant", "content": msg})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.8
            )
            patient_text = response.choices[0].message.content
            
            # Generate patient's voice if requested
            patient_audio = b""
            if include_voice:
                voice = self.select_patient_voice(config)
                patient_audio = self.text_to_speech(patient_text, voice)
            
            return patient_text, patient_audio
            
        except Exception as e:
            return f"I'm having trouble responding right now.", b""
    
    def text_to_speech(self, text: str, voice: str = "alloy") -> bytes:
        """Convert patient's text to speech"""
        try:
            # Remove action descriptions for speech
            clean_text = re.sub(r'\*[^*]*\*', '', text).strip()
            
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=clean_text,
                speed=1.0
            )
            return response.content
        except Exception as e:
            st.error(f"Text-to-speech error: {str(e)}")
            return b""
    
    def select_patient_voice(self, config: PatientConfig) -> str:
        """Select voice for the patient based on their characteristics"""
        if config.gender.lower() == "male":
            if config.age < 30:
                return "echo"  # Younger male
            else:
                return "onyx"  # Mature male
        else:  # Female
            if config.age < 25:
                return "nova"   # Younger female
            elif "Anxiety" in config.diagnosis:
                return "shimmer"  # Softer, anxious
            elif "Depression" in config.diagnosis:
                return "alloy"   # More subdued
            else:
                return "nova"    # Default female
    
    def build_patient_prompt(self, config: PatientConfig, rapport: float, openness: float) -> str:
        """Build prompt for the AI patient"""
        
        core_descriptions = self._traits_to_descriptions(config.core_traits)
        disorder_descriptions = self._disorder_traits_to_descriptions(config.disorder_traits, config.diagnosis)
        
        return f"""You are {config.name}, a {config.age}-year-old {config.gender.lower()} patient in therapy.

DIAGNOSIS: {config.diagnosis}
BACKGROUND: {config.background_story}
SESSION CONTEXT: {config.session_context}

PERSONALITY TRAITS:
{core_descriptions}

DISORDER SYMPTOMS:
{disorder_descriptions}

CURRENT STATE:
- Rapport with therapist: {rapport:.1f}/10
- Openness level: {openness:.1f}/10

IMPORTANT INSTRUCTIONS:
- You are the PATIENT, not the therapist
- Respond to what the therapist says to you
- Stay completely in character as {config.name}
- Show your symptoms through your words and behavior
- Keep responses conversational (2-4 sentences)
- React authentically based on your personality traits
- Don't be artificially cooperative - show realistic resistance when appropriate

Remember: You are experiencing real psychological distress. Respond as a real patient would."""

    def _traits_to_descriptions(self, traits: CoreTraits) -> str:
        descriptions = []
        
        if traits.emotional_intensity > 7:
            descriptions.append("Your emotions are very intense and overwhelming")
        if traits.mood_stability < 3:
            descriptions.append("Your mood changes rapidly and unpredictably")
        if traits.trust_level < 4:
            descriptions.append("You have difficulty trusting others, including therapists")
        if traits.attachment_anxiety > 7:
            descriptions.append("You fear abandonment and rejection intensely")
        if traits.catastrophic_thinking > 7:
            descriptions.append("You tend to imagine worst-case scenarios")
        if traits.self_criticism > 7:
            descriptions.append("You are very hard on yourself and self-critical")
        if traits.verbal_expressiveness < 4:
            descriptions.append("You tend to give short, minimal responses")
        elif traits.verbal_expressiveness > 7:
            descriptions.append("You tend to be very talkative and expressive")
        if traits.defensiveness > 7:
            descriptions.append("You become defensive easily when challenged")
            
        return "- " + "\n- ".join(descriptions) if descriptions else "- Generally typical patterns"
    
    def _disorder_traits_to_descriptions(self, traits: DisorderTraits, diagnosis: str) -> str:
        descriptions = []
        
        if "Borderline" in diagnosis:
            if traits.abandonment_sensitivity > 6:
                descriptions.append("Intense fear of being abandoned or rejected")
            if traits.identity_instability > 6:
                descriptions.append("Uncertain about who you are and what you want")
            if traits.impulsivity > 6:
                descriptions.append("Tendency to act impulsively when distressed")
                
        elif "Depression" in diagnosis:
            if traits.hopelessness > 6:
                descriptions.append("Feeling hopeless about the future")
            if traits.energy_level < 4:
                descriptions.append("Very low energy and motivation")
            if traits.anhedonia > 6:
                descriptions.append("Little interest or pleasure in activities")
                
        elif "Anxiety" in diagnosis:
            if traits.worry_intensity > 6:
                descriptions.append("Constant, intense worrying about many things")
            if traits.physical_anxiety > 6:
                descriptions.append("Physical symptoms of anxiety")
            if traits.perfectionism > 7:
                descriptions.append("Very high standards and fear of making mistakes")
        
        return "- " + "\n- ".join(descriptions) if descriptions else "- Mild symptoms"

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
        
        challenging_impact = technique_scores.get("cbt", 0)
        
        defensiveness_modifier = (10 - patient_traits.defensiveness) / 10
        trust_modifier = patient_traits.trust_level / 10
        
        rapport_change = (positive_impact * 2 - challenging_impact * 0.5) * defensiveness_modifier * trust_modifier
        
        return max(-1.0, min(1.0, rapport_change))

@st.cache_resource
def get_patient_simulator():
    return AIPatientSimulator()

@st.cache_resource  
def get_analyzer():
    return TherapeuticAnalyzer()

def voice_input_component():
    """Create a voice input component using Streamlit components"""
    voice_html = """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 20px; margin: 10px 0; text-align: center;">
        <button id="voiceBtn" onclick="toggleVoiceRecognition()" style="background: #ff4b4b; color: white; border: none; border-radius: 50%; width: 80px; height: 80px; font-size: 28px; cursor: pointer; margin: 10px; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);">🎙️</button>
        <div id="voiceStatus" style="color: white; font-size: 18px; margin: 10px; font-weight: 500;">Click microphone to speak as therapist</div>
        <div id="voiceResult" style="background: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 15px; margin: 10px 0; color: white; min-height: 50px;">Speak naturally - your speech will be sent automatically</div>
        <div style="background: rgba(76, 175, 80, 0.2); border-radius: 8px; padding: 10px; margin: 10px 0; color: white; font-size: 14px;">💡 Auto-send enabled: Patient will respond when you finish speaking</div>
    </div>

    <script>
        let recognition;
        let isListening = false;
        let currentTranscript = '';
        let silenceTimer;
        
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            
            recognition.onstart = function() {
                isListening = true;
                document.getElementById('voiceBtn').style.background = '#00d4aa';
                document.getElementById('voiceBtn').innerHTML = '⏹️';
                document.getElementById('voiceStatus').textContent = 'Listening... speak naturally as the therapist';
                document.getElementById('voiceResult').textContent = 'Listening...';
            };
            
            recognition.onresult = function(event) {
                let finalTranscript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript;
                    }
                }
                
                if (finalTranscript) {
                    currentTranscript = finalTranscript;
                    document.getElementById('voiceResult').textContent = finalTranscript;
                    
                    clearTimeout(silenceTimer);
                    silenceTimer = setTimeout(() => {
                        if (currentTranscript && isListening) {
                            sendVoiceInput();
                        }
                    }, 2000);
                }
            };
            
            recognition.onend = function() {
                if (currentTranscript && isListening) {
                    sendVoiceInput();
                } else {
                    resetVoiceState();
                }
            };
            
            recognition.onerror = function(event) {
                document.getElementById('voiceStatus').textContent = 'Error: ' + event.error + '. Click to try again.';
                resetVoiceState();
            };
        }
        
        function toggleVoiceRecognition() {
            if (!recognition) return;
            
            if (isListening) {
                recognition.stop();
            } else {
                currentTranscript = '';
                recognition.start();
            }
        }
        
        function sendVoiceInput() {
            if (currentTranscript.trim()) {
                document.getElementById('voiceStatus').textContent = 'Processing... Patient will respond shortly';
                document.getElementById('voiceResult').textContent = 'You said: "' + currentTranscript + '" - Sending to patient...';
                
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: currentTranscript.trim()
                }, '*');
                
                resetVoiceState();
            }
        }
        
        function resetVoiceState() {
            isListening = false;
            document.getElementById('voiceBtn').style.background = '#ff4b4b';
            document.getElementById('voiceBtn').innerHTML = '🎙️';
            document.getElementById('voiceStatus').textContent = 'Click microphone to speak as therapist';
            clearTimeout(silenceTimer);
        }
    </script>
    """
    
    return components.html(voice_html, height=200)

def create_audio_player(audio_bytes: bytes, auto_play: bool = True) -> str:
    """Create an auto-playing audio player for patient responses"""
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
    st.title("🧠🎙️ AI Patient Simulator")
    st.markdown("**You are the THERAPIST - Speak naturally and hear the patient respond**")
    
    with st.expander("ℹ️ How This Works", expanded=False):
        st.markdown("""
        **Natural Therapy Conversation:**
        
        **🎙️ Voice Mode:**
        - Click microphone and speak naturally as the therapist
        - Your speech automatically triggers patient response
        - You only HEAR the patient's voice (not your own)
        - Speak freely - no scripted responses needed
        
        **⌨️ Text Mode:**
        - Type what you want to say as the therapist
        - Patient responds with text or voice
        
        **🎯 Practice Goals:**
        - Have natural conversations with AI patients
        - Build rapport through your therapeutic approach
        - Learn how different techniques affect patient responses
        """)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Voice mode toggle
        st.session_state.voice_mode = st.checkbox(
            "🎙️ Enable Voice Mode", 
            value=st.session_state.voice_mode,
            help="Speak naturally + hear patient voice responses"
        )
        
        st.divider()
        
        render_template_selection()
        
        st.divider()
        render_session_controls()
    
    if st.session_state.session_active and st.session_state.patient_config:
        render_therapy_interface()
    else:
        render_welcome_screen()

def render_therapy_interface():
    """Main therapy session interface"""
    
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
    
    # Display conversation
    for i, (role, message) in enumerate(st.session_state.messages):
        with st.chat_message(role):
            if role == "patient":
                processed_message = process_actions(message, st.session_state.show_actions)
                st.markdown(f"**{st.session_state.patient_config.name}**: {processed_message}")
                
                # Auto-play patient audio if available
                if st.session_state.voice_mode and f"patient_audio_{i}" in st.session_state:
                    audio_html = create_audio_player(st.session_state[f"patient_audio_{i}"], auto_play=True)
                    st.markdown(audio_html, unsafe_allow_html=True)
            else:  # therapist
                st.markdown(f"**You (Therapist)**: {message}")
    
    # Input interface
    st.markdown("---")
    
    if st.session_state.voice_mode:
        st.markdown("### 🎙️ Speak Naturally as Therapist")
        st.markdown("*Click microphone, speak, then click 'Send Response' for the patient to reply*")
        
        # Voice input component
        voice_input_component()
        
        # Alternative text input
        st.markdown("**Alternative: Type instead of speaking**")
        text_input = st.text_area(
            "Type your response:",
            placeholder="Or type here if you prefer text...",
            height=80,
            key="text_input_voice_mode"
        )
        
        if st.button("Send Text Response", disabled=not text_input.strip()):
            handle_therapist_response(text_input)
    else:
        # Text-only mode
        st.markdown("### ⌨️ Type Your Response as Therapist")
        if prompt := st.chat_input("What would you like to say to the patient?"):
            handle_therapist_response(prompt)

def handle_therapist_response(therapist_message: str):
    """Process therapist's response and generate patient's reply"""
    
    # Add therapist message to conversation
    st.session_state.messages.append(("therapist", therapist_message))
    
    # Analyze therapeutic techniques
    analyzer = get_analyzer()
    techniques = analyzer.analyze_response(therapist_message)
    rapport_change = analyzer.calculate_rapport_change(techniques, st.session_state.patient_config.core_traits)
    
    # Update rapport and openness
    st.session_state.rapport_level = max(0, min(10, st.session_state.rapport_level + rapport_change))
    st.session_state.patient_openness = max(0, min(10, st.session_state.patient_openness + rapport_change * 0.5))
    
    # Generate patient's response
    simulator = get_patient_simulator()
    patient_response, patient_audio = simulator.generate_patient_response(
        st.session_state.patient_config,
        [msg[1] for msg in st.session_state.messages],
        st.session_state.rapport_level,
        st.session_state.patient_openness,
        include_voice=st.session_state.voice_mode
    )
    
    # Add patient response to conversation
    st.session_state.messages.append(("patient", patient_response))
    
    # Store patient audio
    if patient_audio:
        message_index = len(st.session_state.messages) - 1
        st.session_state[f"patient_audio_{message_index}"] = patient_audio
    
    # Show detected techniques
    detected_techniques = [tech for tech, score in techniques.items() if score > 0]
    if detected_techniques:
        st.info(f"🔍 Detected techniques: {', '.join(detected_techniques)}")
    
    st.rerun()

def process_actions(text, show_actions):
    """Process action descriptions"""
    if show_actions:
        return re.sub(r'\*(.*?)\*', r'***\1***', text)
    else:
        return re.sub(r'\*[^*]*\*', '', text).strip()

def render_template_selection():
    st.subheader("📋 Select Patient")
    
    template_descriptions = {
        "emma_bpd": "**Emma (19, BPD)** - Emotional, fear of abandonment",
        "david_mdd": "**David (45, Depression)** - Low energy, hopeless", 
        "sarah_gad": "**Sarah (28, Anxiety)** - Worried, perfectionist"
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
        st.success(f"Loaded {st.session_state.patient_config.name}!")
        st.rerun()

def render_session_controls():
    st.subheader("🎛️ Session Controls")
    
    st.session_state.show_actions = st.checkbox(
        "Show patient actions", 
        value=st.session_state.show_actions
    )
    
    if not st.session_state.patient_config:
        st.warning("⚠️ Select a patient first")
        return
    
    if not st.session_state.session_active:
        if st.button("▶️ Start Therapy Session", type="primary", use_container_width=True):
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
    """Start therapy session with patient's opening statement"""
    st.session_state.session_active = True
    st.session_state.messages = []
    st.session_state.rapport_level = 5.0
    st.session_state.patient_openness = 3.0
    
    # Generate patient's opening statement
    simulator = get_patient_simulator()
    initial_response, initial_audio = simulator.generate_patient_response(
        st.session_state.patient_config, 
        [], 
        st.session_state.rapport_level, 
        st.session_state.patient_openness,
        include_voice=st.session_state.voice_mode
    )
    
    st.session_state.messages.append(("patient", initial_response))
    
    # Store initial audio
    if initial_audio:
        st.session_state["patient_audio_0"] = initial_audio
    
    st.rerun()

def render_welcome_screen():
    st.markdown("""
    ## Welcome to AI Patient Simulator! 🧠
    
    **You are the THERAPIST having natural conversations with AI patients.**
    
    ### 🎯 How it works:
    
    **👨‍⚕️ YOU = Therapist**
    - Speak or type naturally (no scripts needed!)
    - Ask whatever questions feel right
    - Use your therapeutic instincts
    
    **🤖 AI = Patient** 
    - Responds realistically based on their condition
    - You HEAR their voice (not your own)
    - Reacts authentically to your approach
    
    ### 🚀 Getting Started:
    1. **Select a patient** in the sidebar (Emma, David, or Sarah)
    2. **Enable voice mode** to hear the patient speak
    3. **Start session** - patient will introduce themselves
    4. **Click microphone and speak naturally** - no preparation needed!
    
    ### 🎙️ Voice Experience:
    - **Natural conversation flow** - speak when ready
    - **Auto-send** - patient responds when you finish talking
    - **Patient voices only** - you hear them, not yourself
    - **Browser-based** - works in Chrome, Safari, Edge
    
    **Ready to start a natural therapy conversation?**
    """)

if __name__ == "__main__":
    main()
