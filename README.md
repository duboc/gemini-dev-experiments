## Github Bot Readme

This application utilizes the power of large language models and Github issues to provide an informative and interactive chat experience. It allows users to ask questions about a specific Github repository and receive responses based on the content of its issues.

**Functionality:**

1. **Repository Selection:** Users can input the name of a Github repository in the format "repo/name". 
2. **Issue Loading:** Upon clicking the "Iniciar bot" button, the application fetches and processes the issues from the specified repository.
3. **Chat Interface:** Users can then ask questions related to the repository in a chat-like interface. 
4. **AI-Powered Responses:** The application leverages a large language model (LLM) and information retrieval techniques to provide answers based on the content of the loaded issues. 
5. **Chat History:**  The conversation history is preserved, allowing users to refer back to previous interactions.

**Technical Details:**

* **Libraries:** Streamlit, Langchain, VertexAI
* **LLM:** The application utilizes the `gemini-1.5-pro-preview-0409` model from VertexAI.
* **Embeddings:**  `textembedding-gecko-multilingual@latest` model from VertexAI is used for generating text embeddings.
* **Vector Store:** FAISS is used for efficient storage and retrieval of text embeddings. 
* **Chain:** A RetrievalQA chain is employed to combine the LLM with the vector store for question answering.

**Setup:**

1. **Clone the repository**
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Set Environment Variable:** 
    * Create a Github personal access token with appropriate permissions.
    * Set the environment variable `GITHUB_TOKEN` to your token value.

**Usage:**

1. Run the application: `streamlit run app.py`
2. Enter the desired Github repository name.
3. Click "Iniciar bot" to load and process the issues.
4. Start asking questions in the chat interface. 

**Additional Notes:**

* The application currently focuses on Github issues and does not include pull requests. 
* The response quality depends on the content and quality of the issues in the selected repository.
* The `Reset Demo State` button can be used to clear the chat history and start a new session. 