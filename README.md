# 🧠 InnerRoute: Emotionally Intelligent AI Therapist

InnerRoute is an experimental AI-powered therapeutic chatbot designed to simulate emotionally intelligent, human-like conversations. It leverages large language models (LLMs) and custom prompt engineering to deliver empathetic, context-aware support to users in mental health contexts.

> ⚠️ **Disclaimer**: This is a personal project created for educational and experimental purposes. It is *not* a replacement for professional therapy or mental health services.

---

## 🌟 Project Goals

- Design a conversational agent that feels emotionally attuned and human.
- Explore prompt engineering strategies that guide LLMs to respond with empathy, nuance, and care.
- Build and test an emotion recognition system to adapt responses in real time.
- Create a robust, human-annotated dataset for emotion classification and quality tuning.

---

## 🧩 Key Features

- **Prompt Engineering**: Crafted and refined over multiple iterations to improve naturalness and emotional depth.
- **NLP Pipeline**: Includes sentiment and emotion detection using Transformer-based models.
- **Emotion Recognition Dataset**: Manually labeled text samples based on tone, affect, and psychological context.
- **LLM Integration**: Powered by OpenAI's API with custom-designed prompt strategies.
- **Modular Backend**: Clean Python architecture for plug-and-play emotion models, prompt updates, and evaluations.
- **Evaluation Tools**: Measure performance using CSAT metrics, output quality comparisons, and user feedback logs.

---

## 🧠 Technologies Used

- **Python**, **Pandas**, **OpenAI API**, **Transformers**
- **Jupyter Notebooks** for iterative prototyping
- **Manual Annotation + Labeling Guidelines** for emotion classification
- **Prompt Engineering** using structured iterations and refinement loops
- **N8N** for agent orchestration and automation (external integration)

---

## 📁 Repository Structure

```bash
InnerRoute/
├── prompts/                # Prompt experiments and iterations
├── notebooks/              # Jupyter notebooks for NLP pipeline and testing
├── data/                   # Labeled datasets and guidelines
├── src/                    # Core app and processing scripts
├── docs/                   # Diagrams, results, and presentations
├── evaluations/            # Output logs and performance reports
├── .github/                # GitHub templates
├── requirements.txt        # Python dependencies
└── README.md               # You're here!
