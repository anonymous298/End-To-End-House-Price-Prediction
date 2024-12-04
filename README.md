# **House Price Prediction - End-to-End Project**

This project is an end-to-end Machine Learning application that predicts the house prices based on user inputs. It includes modular code, a Flask-based web application, and is containerized using Docker for easy deployment.


![1](https://github.com/user-attachments/assets/cd39e430-388f-4c18-9776-3dc0d657aed3)

## Performance

![Capture1](https://github.com/user-attachments/assets/e0814a7d-d393-48ef-9665-b57828aa5134)



## Features

- **Machine Learning & Deep Learning**: Predicts house prices using trained DL model.
- **Web Interface**: User-friendly UI built with Flask.
- **Modular Codebase**: Code is organized into reusable modules.
- **Containerization**: Fully containerized using Docker for easy deployment.

## Requirements
To run this project, ensure the following are installed:

- Python 3.9 or above
- Docker
- pip (Python package installer)
## Setup Instructions
Clone the Repository

```bash
https://github.com/anonymous298/End-To-End-House-Price-Prediction.git
cd End-To-End-House-Price-Prediction
```


## Run Locally without Docker

1. Create a Virtual Environment:

```bash
conda create -p venv python==3.10
```

2. Install Dependencies:

```bash
pip install -r requirements.txt
```
3. Run the Application:

```bash
python app.py
```

4. Open the app in your browser:

```bash
http://127.0.0.1:8080
```

## Run Using Docker

1. Build the Docker Image:

```bash
docker build -t house-price-model .
```

2. Run the Docker Container:

```bash
docker run -d -p 8080:8080 house-price-model
```

4. Open the app in your browser:

```bash
http://localhost:8080
```

## Pull and Run Prebuilt Docker Image

Use the prebuild image:

1. Pull the Docker Image:

```bash
docker pull talha213/house-price-model
```

2. Run the Docker Container:

```bash
docker run -d -p 8080:8080 house-price-model
```

## Using the Application
1. Open the web application in your browser:
```bash
http://localhost:8080
```
2. On the homepage, click "Get Started" to navigate to the form.
3. Fill in the details like sqft, total_room, floors, etc.
4. Submit the form to get the prediction result.

## Project Highlights

- Built using a modular coding approach for  scalability.
- Provides an interactive web interface for predictions.
- Easily deployable using Docker.

## **About Me**  

Hi! I'm [Talha](https://github.com/anonymous298), a passionate developer and tech enthusiast focused on building impactful projects. I specialize in **AI/ML**, and crafting efficient solutions for complex problems.  

### **Skills**  
- 🧠 Artificial Intelligence & Machine Learning  
- 💻 Web Development (Frontend & Backend)  
- 📊 Data Analysis & Visualization  

### **Connect with Me**  
- [GitHub](https://github.com/anonymous298)  
- [LinkedIn](https://linkedin.com/in/muhmmad-talha937/)
