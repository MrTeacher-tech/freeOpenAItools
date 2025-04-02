# Free OpenAI Tools

A library for creating free tools for OpenAI agents.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to provide a set of tools and agents that enhance the capabilities of OpenAI models. It includes various agents, utility functions, and a Streamlit-based web application for demonstration purposes.

## Features

- **Agents Module (`agents.py`)**: Contains implementations of different agents that interact with OpenAI models.
- **Streamlit Application (`streamlit_app.py`)**: A web-based interface to interact with the agents and tools.
- **Knowledge Files (`knowledge_files/`)**: Directory containing supplementary data and resources used by the agents.
- **Testing Scripts (`test_agents.py`, `test_freeOpenAITools.py`)**: Unit tests to ensure the reliability of the agents and tools.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/MrTeacher-tech/freeOpenAItools.git


2. **Install required dependencies**

It’s recommended to use a virtual environment:

   ```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
```

Then, install the dependencies:

   ```bash
pip install -r requirements.txt
```

3. **Usage**
Set Up Environment Variables
Create a .env file in the root directory and add your OpenAI API key:

   ```bash
OPENAI_API_KEY=your_openai_api_key
```
Run the Streamlit Application
   ```bash
streamlit run streamlit_app.py
```
This will launch the web application in your default browser.

4. **Project Structure**

freeOpenAItools/
├── knowledge_files/
│   └── ...                 # Supplementary data and resources
├── .gitignore              # Git ignore file
├── agents.py               # Agents interacting with OpenAI models
├── freeOpenAITools.py      # Core library functions
├── requirements.txt        # Project dependencies
├── streamlit_app.py        # Streamlit web application
├── test_agents.py          # Tests for agents module
├── test_freeOpenAITools.py # Tests for core library functions
└── README.md               # Project documentation

5. **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

6. **License**
This project is licensed under the MIT License.


