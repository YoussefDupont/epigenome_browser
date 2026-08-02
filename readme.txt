Epigenomic Network Browser - Getting Started

A quick guide to setting up and running the app locally. Pick the setup method that matches what you already have installed.

Requirements:

- A terminal (Command Prompt / PowerShell on Windows, Terminal on macOS/Linux)
- Python 3.10 or newer or Miniconda/Anaconda
- The project files (clone or download this repository)
- deepTools installed to PATH in your system

Option A - Vanilla Python (venv)

This uses Python's built-in virtual environment tool. Only a clean python install is needed.

    1. Check your Python version

        >python --version

    You need 3.10 or higher. If you see Python 2.x, try python3 --version and use python3 / pip3 throughout.

    2. Clone or download the project

        git clone https://github.com/YoussefDupont/epigenome_browser.git
        cd path/to/folder/epigenome_browser

    Or download the ZIP from GitHub and unzip it, then cd into the folder. If you downloaded and unzipped the repository, your path is going to be what is written in the file explorer address bar when inside the unzipped folder.

    3. Create a virtual environment

        python -m venv venv

    This creates a folder called venv/ that holds an isolated Python install.

    4. Activate the environment
    ***If you're opening the app again for at least the second time, make sure you run the same cd command you did earlier***

        macOS / Linux ->    source venv/bin/activate
        Windows (CMD) ->    venv\Scripts\activate.bat
        Windows (PowerShell) -> venv\Scripts\Activate.ps1
    
    Your terminal prompt will change to show (venv) when it's active.

    5. Install dependencies

        pip install -r requirements.txt

    6. Run the app

        python app.py

        The app will open automatically at http://127.0.0.1:5000 in your browser.

    Deactivating

        When you're done, type deactivate to leave the virtual environment.

Option B - Miniconda / Anaconda (conda)

Conda manages both the environment and the Python version.

    1. Check conda is available

        conda --version

    If not installed, download Miniconda (lightweight) or Anaconda (full suite).

    2. Clone or download the project

        git clone https://github.com/YoussefDupont/epigenome_browser.git
        cd path/to/folder/epigenome_browser

    3. Create a conda environment

        conda create -n epigenome_browser python=3.11 -y

    This creates an environment named epigenome_browser with Python 3.11.

    4. Activate the environment

        conda activate epigenome_browser

    5. Install dependencies

        pip install -r requirements.txt

    6. Run the app

        python app.py

    The app will open automatically at http://127.0.0.1:5000 in your browser.

    Deactivating

        conda deactivate

Using the App

Once the browser opens:

    1. Load Data - upload a Hi-C contact .tsv file and a TAD boundary .bed file, or load a prebuilt network .json.
    2. Select chromosomes (if prompted) - choose which chromosomes to include.
    3. Review column names - the app auto-cleans annotation column names; confirm or edit them.
    4. Configure annotations - pick which histone/RNA-seq/compartment columns to colour nodes by.
    5. Select reference genome - choose the genome assembly for the gene report viewer (default: hg38).
    6. Build the network - adjust the edge weight percentile slider and click Build Network to open the interactive graph in a new tab.

Troubleshooting