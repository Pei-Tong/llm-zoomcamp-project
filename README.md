# movie-assistant
This is your friendly movie assistant, which is a RAG application built as a part of LLM zoomcamp


- Dataset: [BrendanMartin/IMDB-Movie-Data.csv](https://github.com/LearnDataSci/articles/blob/master/Python%20Pandas%20Tutorial%20A%20Complete%20Introduction%20for%20Beginners/IMDB-Movie-Data.csv)
- LLM: [gpt-4o-mini](https://platform.openai.com/docs/models/gpt-4o-mini)
- Embedding model: [multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)

## Prerequisites
- Windows 11
- WSL2 (Windows Subsystem for Linux 2)
- Ubuntu (In WSL, the version may vary depending on the installation)
- Python 3.12 or higher
- Docker
- Conda or Miniconda

## Install and Set Up

1. Male sure WSL has been installed and updated to the newest version of Ubuntu. You cna run below's commend to check the version of Ubuntu.
   ```
   wsl -l -v
   ```

2. If you need to update Ubuntu，please take Microsoft official document as reference.

3. Clone this repository in WSL Ubuntu terminal
    ```
    git clone https://github.com/your-username/movie-assistant.git
    cd movie-assistant
    ```

4. Install Conda（If you don't have one）.You can download the installation script for Linux from the Miniconda website.


5.Use `environment.yml` create a new conda environment
    ```
    conda env create -f environment.yml
    ```

6.Activate the environment
    ```
    conda activate movie-assistant
    ```

## run docker
Run Elasticsearch in WSL Ubuntu using the following Docker commands:

bash
docker run -it \
    --rm \
    --name elasticsearch \
    -m 4GB \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.15.0

Note: Make sure you have installed Docker in WSL Ubuntu, if not, follow the official Docker documentation.

## Run the Application

1. Make sure you've activated Conda environment：
   ```
   conda activate movie-assistant
   ```

2. Run Streamlit：
   ```
   streamlit run notebooks/elastic.py
   ```

3. Open the displayed URL in the browser on the Windows host. (usually http://localhost:8501)

Warning: This project has only been tested on WSL2-Ubuntu. Compatibility with other operating systems is not guaranteed.

## Dataset

Dataset: [BrendanMartin/IMDB-Movie-Data.csv](https://github.com/LearnDataSci/articles/blob/master/Python%20Pandas%20Tutorial%20A%20Complete%20Introduction%20for%20Beginners/IMDB-Movie-Data.csv)

## How to execute it
1. Type something about the movie you're searching, including title, genre, or description.
2. When there is the fit movie, it can provide some information about "Title, Genre, Description, Director, Actors, Year, Runtime (Minutes), Rating, Votes, Revenue (Millions), Metascore"

## Below is a preview of the application interface:
![Preview](.Preview.png)


## Trouble shoot
If you encounter any problems:
1. Make sure you've set up the OPENAI_API_KEY correctly.
2. Check Elasticsearch is running and can be accessed.
3. Make sure every required dependices are installed correctly.

## Contribution





