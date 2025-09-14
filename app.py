# app.py - AI Patient Simulator for Streamlit Cloud
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

# Pre-built templates
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

# Therapeutic analyzer
class TherapeuticAnalyzer:
    THERAPEUTIC_TECHNIQUES = {
        "validation": ["understand", "makes sense", "hear you", "valid", "difficult"],
        "empathy": ["feel", "sounds", "imagine", "must be", "experiencing"],
        "clarification": ["what do you mean", "can you tell me more", "help me understand", "clarify"],
        "reflection": ["you're saying", "sounds like", "it seems", "you feel", "you're experiencing"],
        "rapport": ["thank you for sharing", "appreciate", "brave", "strength", "trust"],
        "cbt": ["thought", "thinking", "evidence", "alternative", "realistic", "challenge"],
        "acceptance": ["okay", "that's alright", "understandable", "human", "normal"],
        "acknowledgment": ["I see", "I notice", "I hear", "yes", "right"],
        "cognitive_restructuring": ["different way", "perspective", "reframe", "consider", "another view"]
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
            technique_scores.get("acceptance", 0) * 0.2 +
            technique_scores.get("acknowledgment", 0) * 0.1
        )
        
        challenging_impact = technique_scores.get("cbt", 0) + technique_scores.get("cognitive_restructuring", 0)
        
        defensiveness_modifier = (10 - patient_traits.defensiveness) / 10
        trust_modifier = patient_traits.trust_level / 10
        
        rapport_change = (positive_impact * 2 - challenging_impact * 0.5) * defensiveness_modifier * trust_modifier
        
        return max(-1.0, min(1.0, rapport_change))

# Patient simulator
class OpenAIPatientSimulator:
    def __init__(self):
        self.client = get_openai_client()
    
    def generate_patient_response(self, config: PatientConfig, conversation_history: List[str], 
                                rapport: float, openness: float) -> str:
        system_prompt = self.build_system_prompt(config, rapport, openness)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        for i, msg in enumerate(conversation_history[-6:]):
            role = "assistant" if i % 2 == 0 else "user"
            messages.append({"role": role, "content": msg})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.7 + (random.random() - 0.5) * 0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I'm having trouble responding right now. Please try again."
    
    def build_system_prompt(self, config: PatientConfig, rapport: float, openness: float) -> str:
        core_descriptions = self._traits_to_descriptions(config.core_traits)
        disorder_descriptions = self._disorder_traits_to_descriptions(config.disorder_traits, config.diagnosis)
        rapport_desc = self._get_rapport_description(rapport)
        openness_desc = self._get_openness_description(openness)
        
        return f"""You are {config.name}, a {config.age}-year-old {config.gender.lower()} patient in therapy.

DIAGNOSIS: {config.diagnosis}
BACKGROUND: {config.background_story}
SESSION CONTEXT: {config.session_context}

PERSONALITY TRAITS:
{core_descriptions}

DISORDER-SPECIFIC SYMPTOMS:
{disorder_descriptions}

CURRENT EMOTIONAL STATE:
- Rapport with therapist: {rapport_desc}
- Openness level: {openness_desc}

RESPONSE GUIDELINES:
1. Stay completely in character as {config.name}
2. Respond naturally as a real patient would
3. Show symptoms through behavior, not by stating them directly
4. Let your traits influence your communication style
5. React authentically to the therapist's approach
6. Sometimes have mood shifts or emotional reactions
7. Don't be artificially cooperative - show realistic resistance or confusion when appropriate
8. Keep responses conversational length (2-4 sentences typically)

Remember: You are not playing a role for educational purposes - you ARE {config.name} experiencing these struggles."""

    def _traits_to_descriptions(self, traits: CoreTraits) -> str:
        descriptions = []
        
        if traits.emotional_intensity > 7:
            descriptions.append("Your emotions are very intense and overwhelming")
        elif traits.emotional_intensity < 3:
            descriptions.append("You tend to feel emotionally numb or disconnected")
            
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
            
        return "- " + "\n- ".join(descriptions) if descriptions else "- Generally typical emotional and social patterns"
    
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
                descriptions.append("Little interest or pleasure in activities you used to enjoy")
                
        elif "Anxiety" in diagnosis:
            if traits.worry_intensity > 6:
                descriptions.append("Constant, intense worrying about many things")
            if traits.physical_anxiety > 6:
                descriptions.append("Physical symptoms of anxiety (tension, racing heart, etc.)")
            if traits.perfectionism > 7:
                descriptions.append("Very high standards and fear of making mistakes")
        
        return "- " + "\n- ".join(descriptions) if descriptions else "- Mild or well-managed symptoms"
    
    def _get_rapport_description(self, rapport: float) -> str:
        if rapport >= 8: return "Strong trust and connection with therapist"
        elif rapport >= 6: return "Growing trust, becoming more comfortable"  
        elif rapport >= 4: return "Neutral, cautiously engaging"
        elif rapport >= 2: return "Guarded, some mistrust"
        else: return "Very guarded, resistant, or hostile"
    
    def _get_openness_description(self, openness: float) -> str:
        if openness >= 8: return "Very open, sharing freely and deeply"
        elif openness >= 6: return "Becoming more open, willing to share"
        elif openness >= 4: return "Somewhat open, sharing surface-level information"  
        elif openness >= 2: return "Guarded, minimal sharing"
        else: return "Very closed off, resistant to sharing"

@st.cache_resource
def get_patient_simulator():
    return OpenAIPatientSimulator()

@st.cache_resource  
def get_analyzer():
    return TherapeuticAnalyzer()

def process_actions(text, show_actions):
    if show_actions:
        return re.sub(r'\*(.*?)\*', r'***\1***', text)
    else:
        return re.sub(r'\*[^*]*\*', '', text).strip()

def export_transcript():
    if not st.session_state.messages:
        return None
    
    transcript = []
    transcript.append(f"AI Patient Simulator - Session Transcript")
    transcript.append(f"Patient: {st.session_state.patient_config.name}")
    transcript.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    transcript.append("="*50)
    
    for role, message in st.session_state.messages:
        speaker = st.session_state.patient_config.name if role == "patient" else "Therapist"
        transcript.append(f"{speaker}: {message}")
        transcript.append("")
    
    return "\n".join(transcript)

def main():
    st.title("🧠 AI Patient Simulator")
    st.markdown("**Practice your clinical skills with realistic AI patients**")
    
    with st.expander("ℹ️ About This Demo", expanded=False):
        st.markdown("""
        This is a demo of an AI Patient Simulator for psychology training.
        
        **Features:**
        - Realistic patient responses based on psychological traits
        - Real-time rapport tracking
        - Therapeutic technique detection
        - Custom patient configuration
        
        **How to use:**
        1. Select or create a patient in the sidebar
        2. Start a therapy session
        3. Practice your therapeutic skills
        4. Watch how your approach affects patient rapport
        """)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        setup_mode = st.radio(
            "Setup Method:",
            ["Pre-built Templates", "Custom Configuration"]
        )
        
        if setup_mode == "Pre-built Templates":
            render_template_selection()
        else:
            render_custom_configuration()
        
        st.divider()
        render_session_controls()
    
    if st.session_state.session_active and st.session_state.patient_config:
        render_chat_interface()
    else:
        render_welcome_screen()

def render_template_selection():
    st.subheader("📋 Pre-built Patients")
    
    template_descriptions = {
        "emma_bpd": "**Emma (19, BPD)**: College student, recent breakup, emotional intensity, fear of abandonment",
        "david_mdd": "**David (45, Depression)**: Unemployed executive, low energy, hopelessness, minimal responses", 
        "sarah_gad": "**Sarah (28, Anxiety)**: New mother, perfectionist, constant worry, seeks reassurance"
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

def render_custom_configuration():
    st.subheader("🎛️ Custom Patient")
    
    with st.expander("Basic Information", expanded=True):
        name = st.text_input("Name", "Alex")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 16, 80, 25)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
        
        diagnosis = st.selectbox("Diagnosis", [
            "Borderline Personality Disorder",
            "Major Depressive Disorder", 
            "Generalized Anxiety Disorder"
        ])
    
    with st.expander("Core Personality Traits"):
        emotional_intensity = st.slider("Emotional Intensity", 0.0, 10.0, 5.0, 0.5)
        trust_level = st.slider("Trust Level", 0.0, 10.0, 5.0, 0.5)
        defensiveness = st.slider("Defensiveness", 0.0, 10.0, 5.0, 0.5)
        verbal_expressiveness = st.slider("Verbal Expressiveness", 0.0, 10.0, 5.0, 0.5)
    
    disorder_traits = {}
    with st.expander("Disorder-Specific Traits"):
        if "Borderline" in diagnosis:
            disorder_traits['abandonment_sensitivity'] = st.slider("Abandonment Sensitivity", 0.0, 10.0, 5.0, 0.5)
            disorder_traits['identity_instability'] = st.slider("Identity Instability", 0.0, 10.0, 5.0, 0.5)
            disorder_traits['impulsivity'] = st.slider("Impulsivity", 0.0, 10.0, 5.0, 0.5)
        elif "Depression" in diagnosis:
            disorder_traits['hopelessness'] = st.slider("Hopelessness", 0.0, 10.0, 5.0, 0.5)
            disorder_traits['energy_level'] = st.slider("Energy Level", 0.0, 10.0, 5.0, 0.5)
            disorder_traits['anhedonia'] = st.slider("Loss of Interest", 0.0, 10.0, 5.0, 0.5)
        elif "Anxiety" in diagnosis:
            disorder_traits['worry_intensity'] = st.slider("Worry Intensity", 0.0, 10.0, 5.0, 0.5)
            disorder_traits['perfectionism'] = st.slider("Perfectionism", 0.0, 10.0, 5.0, 0.5)
    
    with st.expander("Background Information"):
        background = st.text_area("Background Story", f"A {age}-year-old {gender.lower()} with {diagnosis.lower()}")
        context = st.text_area("Session Context", "First session, patient appears nervous about being here")
    
    if st.button("Create Custom Patient", type="primary"):
        core_traits = CoreTraits(
            emotional_intensity=emotional_intensity,
            trust_level=trust_level,
            defensiveness=defensiveness,
            verbal_expressiveness=verbal_expressiveness
        )
        
        disorder_traits_obj = DisorderTraits(**disorder_traits)
        
        st.session_state.patient_config = PatientConfig(
            name=name, age=age, gender=gender, diagnosis=diagnosis,
            background_story=background, session_context=context,
            core_traits=core_traits, disorder_traits=disorder_traits_obj
        )
        
        st.session_state.messages = []
        st.session_state.rapport_level = 5.0
        st.session_state.patient_openness = 3.0
        st.session_state.session_active = False
        st.success(f"Created {name}!")
        st.rerun()

def render_session_controls():
    st.subheader("🎛️ Session Controls")
    
    st.session_state.show_actions = st.checkbox(
        "Show action descriptions (*like this*)", 
        value=st.session_state.show_actions
    )
    
    if not st.session_state.patient_config:
        st.warning("⚠️ Configure a patient first")
        return
    
    if not st.session_state.session_active:
        if st.button("▶️ Start Session", type="primary", use_container_width=True):
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
    
    if st.session_state.messages:
        transcript = export_transcript()
        st.download_button(
            "📄 Export Transcript",
            transcript,
            f"session_{st.session_state.patient_config.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            "text/plain",
            use_container_width=True
        )

def start_session():
    st.session_state.session_active = True
    st.session_state.messages = []
    st.session_state.rapport_level = 5.0
    st.session_state.patient_openness = 3.0
    
    simulator = get_patient_simulator()
    initial_response = simulator.generate_patient_response(
        st.session_state.patient_config, [], 
        st.session_state.rapport_level, 
        st.session_state.patient_openness
    )
    
    st.session_state.messages.append(("patient", initial_response))
    st.rerun()

def render_welcome_screen():
    st.markdown("""
    ## Welcome to the AI Patient Simulator! 👋
    
    This tool helps psychology students practice their clinical skills with realistic AI patients.
    
    ### How to get started:
    1. **Configure a patient** in the sidebar (templates or custom)
    2. **Start a session** to begin the simulation
    3. **Practice your therapeutic techniques** and see real-time feedback
    
    ### Features:
    - 🎭 **Realistic patient responses** based on psychological traits
    - 📊 **Real-time rapport tracking** shows how your approach affects the patient
    - 🔍 **Technique detection** identifies therapeutic methods you're using
    - 📝 **Session transcripts** for review and learning
    
    Ready to practice? Configure a patient in the sidebar to begin!
    """)
    
    if st.session_state.patient_config:
        st.info(f"✅ {st.session_state.patient_config.name} is loaded and ready. Click 'Start Session' in the sidebar!")

def render_chat_interface():
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**Patient:** {st.session_state.patient_config.name} ({st.session_state.patient_config.age}, {st.session_state.patient_config.diagnosis})")
        with col2:
            st.metric("Rapport", f"{st.session_state.rapport_level:.1f}/10")
        with col3:
            st.metric("Openness", f"{st.session_state.patient_openness:.1f}/10")
    
    st.divider()
    
    for role, message in st.session_state.messages:
        with st.chat_message(role):
            processed_message = process_actions(message, st.session_state.show_actions)
            if role == "patient":
                st.markdown(f"**{st.session_state.patient_config.name}**: {processed_message}")
            else:
                st.markdown(f"**Therapist**: {processed_message}")
    
    if prompt := st.chat_input("Type your response as the therapist..."):
        handle_therapist_response(prompt)

def handle_therapist_response(message: str):
    st.session_state.messages.append(("therapist", message))
    
    analyzer = get_analyzer()
    techniques = analyzer.analyze_response(message)
    rapport_change = analyzer.calculate_rapport_change(techniques, st.session_state.patient_config.core_traits)
    
    st.session_state.rapport_level = max(0, min(10, st.session_state.rapport_level + rapport_change))
    st.session_state.patient_openness = max(0, min(10, st.session_state.patient_openness + rapport_change * 0.5))
    
    simulator = get_patient_simulator()
    patient_response = simulator.generate_patient_response(
        st.session_state.patient_config, 
        [msg[1] for msg in st.session_state.messages],
        st.session_state.rapport_level, 
        st.session_state.patient_openness
    )
    
    st.session_state.messages.append(("patient", patient_response))
    
    detected_techniques = [tech for tech, score in techniques.items() if score > 0]
    if detected_techniques:
        st.info(f"🔍 Detected techniques: {', '.join(detected_techniques)}")
    
    st.rerun()

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
