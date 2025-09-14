# AI Patient Simulator

An open-source project to create **AI-powered patient simulations** that help psychology and mental health students practice their **clinical interviewing and risk assessment skills** in a safe, controlled environment.

Just like flight simulators train pilots, this project aims to provide a **practice space for therapists-in-training**, especially for **uncommon or high-risk cases** they wouldn’t normally encounter early in training.

---

## 🚀 Vision
- Support future clinicians with **realistic, interactive patient simulations**.
- Cover **high-risk and uncommon cases**: suicide risk, psychosis, trauma, substance withdrawal, abuse disclosures, cultural syndromes, etc.
- Provide **structured feedback and grading** to help students grow.
- Ensure **ethical, safe, and educational use** (training only, not real therapy).

---

## 🧩 Core Components
- **Scenario Engine (State Graphs)**  
  Defines hidden patient states, symptoms, and transitions triggered by student actions.

- **Retrieval-Augmented Generation (RAG)**  
  Keeps each case grounded in consistent patient history, intake notes, and collateral data.

- **Risk & Safety Tools**  
  Modules for suicide risk, abuse suspicion, mandated reporting, and more.

- **Evaluator/Grader**  
  Scores sessions using OSCE-style rubrics (e.g., empathy, alliance, coverage of key domains).

- **Frontend**  
  Chat-based UI (with optional voice), risk indicators, and a post-session debrief.

---

## 🛠️ Tech Stack (MVP)
- **Backend**: Python, FastAPI (or Flask)
- **LLM Orchestration**: LangChain / LiteLLM
- **Vector Store**: Weaviate, Pinecone, or FAISS
- **Frontend**: React (chat + dashboard)
- **Voice (optional)**: WebRTC + TTS/ASR (e.g., OpenAI Whisper, ElevenLabs)

---

## 📚 Roadmap
**Phase 1 (MVP)**  
- Build 6 seed cases with branching states.  
- Implement orchestrator (state + LLM + RAG).  
- Add basic grading and risk-screen tools.  
- Release a simple web UI.  

**Phase 2**  
- Expand to 20+ cases.  
- Add stochastic realism (distractibility, tangents, guardedness).  
- Begin collecting transcripts for fine-tuning adapters.  

**Phase 3**  
- Light fine-tuning (LoRA/adapters) for persona stability and safety reflexes.  
- RLHF on supervisor preferences.  
- Publish validation results (student outcomes, realism scores).  

---

## 🔒 Ethics & Safety
- Training use only — **not a replacement for therapy or real patients**.  
- All cases are **synthetic or anonymized**.  
- Built with guardrails against unsafe or misleading outputs.  
- Requires institutional approval for research use.

---

## 🤝 Contributing
We welcome:
- **Clinical case designers** (psychologists, psychiatrists, social workers).  
- **AI/ML engineers** (NLP, RAG, fine-tuning, safety tools).  
- **Educators** (to define rubrics, learning objectives, and curriculum integration).  

To contribute:
1. Fork this repo  
2. Create a branch (`feature/my-case`)  
3. Commit your changes  
4. Open a Pull Request  

---

## 📄 License
[MIT License](LICENSE) — open for research and educational use.

---

## 🌟 Acknowledgements
Inspired by OSCE simulations, flight simulators, and the vision of making mental health training **more accessible, realistic, and safe**.
