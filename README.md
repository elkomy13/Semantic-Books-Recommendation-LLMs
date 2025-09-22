## Semantic Book Recommender Dashboard

based on user-inputted descriptions, categories, and emotional tones. It leverages natural language processing (NLP) and vector search to match user queries with book descriptions from a preprocessed dataset. The system is built using LangChain for document processing, Hugging Face embeddings for semantic understanding, and Chroma vector database for efficient similarity search. The user interface is powered by Gradio, offering an interactive dashboard where users can explore up to 16 book recommendations displayed in a gallery format with book covers, titles, authors, and truncated descriptions.

The system uses a dataset of book metadata (books_with_emotions.csv) that includes emotional scores (e.g., joy, sadness, anger) for each book, enabling tone-based filtering. The book descriptions are stored in a tagged format (tagged_description.txt) and processed into embeddings for semantic search.

## Models and Technologies Used





- **LangChain**: A framework for building applications with language models, used here for document loading (TextLoader), text splitting (CharacterTextSplitter), and integration with embeddings and vector stores.



- **Hugging Face Embeddings**: Utilizes the sentence-transformers/all-MiniLM-L6-v2 model, a lightweight transformer-based model optimized for sentence-level embeddings. It converts book descriptions into dense vectors for semantic similarity comparisons.



- **Chroma Vector Database**: An open-source vector store used to index and query book description embeddings, enabling fast and accurate similarity searches based on user queries.



- **Gradio**: A Python library for creating web-based user interfaces, used to build the interactive dashboard with input fields for queries, category/tone dropdowns, and a gallery for displaying recommendations.


The system performs semantic search to retrieve books based on query similarity, filters results by category and emotional tone (if specified), and sorts recommendations by emotional scores when a tone is selected (e.g., sorting by "joy" for a "Happy" tone).

## Prerequisites

- Python 3.10 or higher
- Access to a Hugging Face API key (for embeddings)
- The following files (generated from the provided Jupyter notebook or equivalent data preparation):
  - `books_with_emotions.csv`: Book metadata with emotions.
  - `tagged_description.txt`: Tagged book descriptions for vector embedding.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
    ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   venv\Scripts\activate  # On Windows
    ```
3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ``` 
4. **Set up environment variables**:
   - Create a `.env` file in the project root directory.
   - Add your Hugging Face API key:
     ```
     HUGGING_FACE_KEY="your_hugging_face_api_key"
     ```
5. **Prepare the data files**:
   - In Gradio_dashboard.py, replace the hardcoded paths (e.g., G:\LLM Tasks\Books Recommendation\books_with_emotions.csv) with relative or absolute paths to your local files.
   - Similarly, update the path for tagged_description.txt and the placeholder image (cover-not-found.jpg).
6. **Run the Gradio dashboard**:
   ```bash
   python Gradio_dashboard.py
   ```
   - This will launch the Gradio interface locally (usually at http://127.0.0.1:7860/).
   - Open the URL in your browser to interact with the dashboard.
7. **Using the Dashboard**:
   - Input a book description in the text box.
   - Select a category and emotional tone from the dropdown menus.
   - Click the "Get Recommendations" button to receive book suggestions based on your input.
   - The recommended books will be displayed with their titles, authors, categories, emotional tones, and cover images.

## Data Preparation
If you need to regenerate the data:

- Run the provided Jupyter notebook (semantic-book-recommender-using-llms.ipynb).
- It loads raw book data, processes emotions using a model (not specified in the code snippet), and outputs books_with_emotions.csv and tagged_description.txt.
- Ensure you have the necessary libraries installed in your Jupyter environment.

## Notes
- The embeddings model (sentence-transformers/all-MiniLM-L6-v2) is downloaded automatically on first run.
- Ensure your Hugging Face key has access to the required models.
- For production, consider containerizing with Docker or deploying on Hugging Face Spaces/Gradio share.
- If you encounter path issues, make the paths configurable via environment variables or command-line arguments.
