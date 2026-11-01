# 🤖 Multi-Tool Autonomous AI Research Agent

An AI-powered research assistant that uses an LLM and multiple tools to help users research topics, perform calculations, and extract information from uploaded documents through a conversational interface.

Built with **Python, LangChain, Groq API, and Gradio**, this project demonstrates how an AI agent can use external tools to handle different tasks instead of relying only on text generation.

## ✨ Overview

The Multi-Tool Autonomous AI Research Agent is designed to make information gathering and everyday research tasks easier.

Users can interact with the agent through a chat interface. Based on the request and the available tool-calling logic, the agent can select and use relevant tools to perform tasks and generate responses.

## 🚀 Key Features

- **AI-Powered Conversations:** Interact with an LLM through a conversational interface.
- **Web Search:** Search the web to gather information and support research tasks.
- **Python Calculator:** Perform mathematical calculations and solve numerical queries.
- **Document Reading:** Upload supported documents and extract their text for analysis.
- **Multi-Tool Integration:** Use different tools to handle different types of user requests.
- **Autonomous Tool Selection:** The agent can select relevant tools based on the user's request, subject to its implemented tool-calling logic.
- **Interactive Gradio Interface:** Use a simple chat-based interface to communicate with the agent.
- **Groq API Integration:** Generate AI responses using a Groq-hosted language model.

## 🛠️ Technology Stack

| Technology                    | Purpose                                             |
| ----------------------------- | --------------------------------------------------- |
| Python                        | Core programming language                           |
| LangChain                     | Agent and tool integration                          |
| Groq API                      | LLM inference                                       |
| Gradio                        | Interactive user interface                          |
| DuckDuckGo / Tavily           | Web search, depending on configured search provider |
| Python                        | Calculator functionality                            |
| Document processing libraries | Document text extraction                            |
| Git & GitHub                  | Version control and project hosting                 |

## 🏗️ Project Architecture

The application follows a tool-based AI agent architecture.

```text
              USER
                |
                v
       GRADIO CHAT INTERFACE
                |
                v
          AI AGENT
                |
                v
       GROQ LANGUAGE MODEL
                |
                v
       TOOL SELECTION LOGIC
                |
       +--------+---------+
       |        |         |
       v        v         v
   WEB SEARCH CALCULATOR DOCUMENT
                         READER
       |        |         |
       +--------+---------+
                |
                v
        TOOL RESULTS
                |
                v
       AI-GENERATED RESPONSE
                |
                v
              USER
```

## 📁 Project Structure

```text
ai-research-agent/
│
├── main.py              # Main application and Gradio interface
├── tools.py             # Tool definitions and integrations
├── style.css            # Custom interface styling
├── .env                 # API keys and environment variables
├── .gitignore           # Files excluded from Git
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

*The structure above describes the main project files; additional files may be present in your repository.*

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Multi-Tool-Autonomous-AI-Agent.git
```

### 2. Navigate to the Project Directory

```bash
cd Multi-Tool-Autonomous-AI-Agent
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the project root directory:

```env
GROQ_API_KEY=your_groq_api_key
```

Add any additional search-provider API keys required by your configuration.

**Important:** Never upload your `.env` file or expose API keys in your GitHub repository.

### 6. Run the Application

```bash
python main.py
```

Open the local Gradio URL displayed in your terminal to start interacting with the AI agent.

## 💡 Example Use Cases

You can interact with the agent using requests such as:

**Web Research**

> Search for recent developments in artificial intelligence and summarize the key findings.

**Mathematical Calculations**

> Calculate the compound interest on 50,000 at 8% annually for 3 years.

**Document Analysis**

> Upload a PDF document and extract its important information.

**General Research**

> Explain how large language models work and summarize their main applications.

*The exact capabilities depend on the tools configured and enabled in the application.*

## 🔐 Security Considerations

- Store API keys in environment variables.
- Keep `.env` out of version control.
- Avoid uploading confidential or sensitive documents.
- Validate uploaded files before processing.
- Do not treat AI-generated answers or web search results as guaranteed accurate.

## ⚠️ Current Limitations

- Responses depend on the selected language model and available API services.
- Web searbch results depend on the configured search provider.
- Document analysis is limited by supported file formats and text extraction capabilities.
- Large documents and lengthy prompts may encounter model token or API rate limits.
- AI-generated responses may contain inaccurate information and should be verified when accuracy is important.

## 🔮 Future Improvements

- Add more specialized tools and integrations.
- Improve document understanding and long-document processing.
- Implement Retrieval-Augmented Generation (RAG).
- Add conversation memory and context management.
- Improve error handling and agent observability.
- Deploy the application as a publicly accessible web application.

## 🎯 Project Objective

The objective of this project is to explore how LLM-powered agents can interact with multiple tools, make tool-selection decisions, and perform practical tasks through a single conversational interface.

It also provides hands-on experience with AI agents, LLM integration, tool calling, Python development, and interactive application development.

## 👩‍💻 Author

**Maria Rauf**

BS Information Technology Graduate | AI & Python Developer

- GitHub: [Maria-Chaudhary](https://github.com/Maria-Chaudhary)
- LinkedIn: [Add your LinkedIn profile](https://www.linkedin.com/in/maria-rauf-925258279/)

---

⭐ If you find this project interesting, consider giving the repository a star!
