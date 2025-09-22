import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from langchain_community.document_loaders import TextLoader # this convert the raw of text into langchain format
from langchain_text_splitters import CharacterTextSplitter # split the texts into chunks
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma # open source vector database
from huggingface_hub import login
import re
import os
HUGGING_FACE_KEY = os.getenv("HUGGING_FACE_KEY")
login(HUGGING_FACE_KEY)
books=pd.read_csv(r'G:\LLM Tasks\Books Recommendation\books_with_emotions.csv')

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800" # resize the image to larger size
books["large_thumbnail"] = np.where(books["thumbnail"].isnull(), r"G:\LLM Tasks\Books Recommendation\cover-not-found.jpg", books["large_thumbnail"]) # if no image, then use placeholder image

raw_doc = TextLoader(
    r"G:\LLM Tasks\Books Recommendation\tagged_description.txt",
    encoding="utf-8"
).load()

text_splitter=CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    separator="\n"
)

documents=text_splitter.split_documents(raw_doc)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

dp_books=Chroma.from_documents(documents,embeddings)

#in this funaction i want to retrieve all books based on the query then get the isbn13 from the retrieved documents 
# and filter the books dataframe to only include the recommended books after that i want to filter the books based on the category and tone if provided
# finally return the top k books based on the tone if provided
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    books_list = []
    recs = dp_books.similarity_search(query, k=initial_top_k)
    print("Retrieved documents:", len(recs))   # 👈 debug
    for i in range(len(recs)):
        raw_isbn = recs[i].page_content.strip('""').split()[0]
        clean_isbn = re.sub(r"\D", "", raw_isbn)
        print("Extracted ISBN:", clean_isbn)   # 👈 debug
        books_list.append(clean_isbn)
        
    book_recs = books[books["isbn13"].astype(str).isin(books_list)].head(initial_top_k)
    print("Matched books:", len(book_recs))   # 👈 debug
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


# Gradio interface and i want to return the book recommendations and split the description to only show the first 30 words because the grid card has limited space
# and add ... at the end of the description to indicate that there is more text
# also format the authors to show only the first two authors and if there are more than two authors then show all authors with commas and before the last author
# Example: ["J.K. Rowling", "John Tiffany", "Jack Thorne"] → Rowling, John and Jack Thorne
# finally return the image and under the image show the title, authors and description
def recommend_books(
        
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


# Define categories and tones for dropdown menus

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


# Create Gradio interface by choosing a theme "gr.themes.Glass()" and adding a title of the dashboard
# and adding a row with a textbox for user query, dropdown for category and tone and a submit button
# then adding a gallery to show the recommended books with 8 columns and 2 rows because we want to show 16 books at a time
# finally linking the submit button to the recommend_books function and passing the user inputs to the function and displaying the output in the gallery
with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()
