# Teacher Assistant
This project aims to address meaning events in Turkish education by employing the Socratic teaching style. It is designed to function as an interactive teaching assistant that engages students effectively. The project includes tools for uploading and storing a series of PDF documents in a database.

## Install Required Libraries

pip install -r requirements.txt

## Set Environment Variables

Create a .env file and define the necessary environment variables:

OPENAI_API_KEY= 'your_openai_api_key'

## For Starting

Write 'python main.py' to the terminal.

## Implementation Details
The project consists of the following key components:

PDF Document Loader: Utilizes the langchain library to load and process PDF documents from a specified directory.
Text Splitting: Implements a text splitting mechanism using RecursiveCharacterTextSplitter to break down the content into manageable chunks.
Database Integration: Employs chromadb to create a persistent collection for storing the loaded documents and their metadata.
OpenAI API Integration: Interacts with the OpenAI API to provide a Socratic teaching experience, asking questions and facilitating discussions based on the meaning events in the Turkish language.


## Possible Improvements
While the current implementation provides a solid foundation, there are several potential improvements to consider:

Enhanced Error Handling: Implement more robust error handling throughout the document loading and database storage processes to handle potential issues gracefully.
User Interface Development: Consider building a graphical user interface (GUI) to improve user interaction and make it easier for educators to upload and manage documents.
Feedback Mechanism: Incorporate a feedback mechanism for students to evaluate their learning experience, allowing for continuous improvement of the teaching approach.
