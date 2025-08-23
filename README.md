# About the project
This project is a FastAPI application that allows users to chat with a PDF document. It utilizes a sophisticated Retrieval-Augmented Generation (RAG) architecture to handle large files efficiently. The workflow is as follows: first, the user uploads a PDF, which the application processes using PyMuPDF to extract the text. This text is then broken down into smaller chunks, which are converted into numerical representations (embeddings) using the Google Gemini API. These embeddings are stored in a high-performance FAISS vector store. When a user asks a question, the application searches this store to find and retrieve only the most relevant text chunks from the document. Finally, these retrieved chunks, along with the original question, are sent to the Gemini model to generate a precise, contextual, and accurate answer based solely on the provided document content.

# Run local
uvicorn app:app --reload

# Example
### Using gemini to upload a pdf file
![alt text](image.png)

### Using the id to ask about the document
![alt text](image-1.png)

### The answer in portuguese
![alt text](image-2.png)

### And answer in english
![alt text](image-3.png)

### Asking the resume
![alt text](image-4.png)
![alt text](image-5.png)

@clarcolaco - 2025