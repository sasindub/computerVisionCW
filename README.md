Computer Vision Coursework

computerVisionCW/
│
├── BE/                   # Backend folder
│   ├── model.h5          # Trained ML model
│   ├── main.py           # FastAPI backend code
│   └── requirements.txt   # Python dependencies
│
└── FE/                   # Frontend folder
    └── predict.html      # Simple HTML interface to upload and predict images

Setup Instructions
1. Clone the Repository
git clone 
cd computerVisionCW

2. Setup Backend
Navigate to the backend folder:
cd BE

Create a virtual environment (optional but recommended):
python -m venv venv

Activate the virtual environment:
Windows:
venv\Scripts\activate

Linux/Mac:
source venv/bin/activate

Install required packages:
pip install -r requirements.txt

fastapi==0.111.0
uvicorn[standard]==0.23.1
tensorflow==2.13.0
numpy==1.26.0
pillow==10.1.0
python-multipart==0.0.6


Run the FastAPI server:
uvicorn main:app --reload

The server will start at http://127.0.0.1:8000.
3. Test Frontend
Open the frontend HTML file:
cd ../FE
open predict.html   # Or just double-click the file

Upload a sample image using the HTML interface. The frontend will send the image to the backend and display the prediction.
Notes

Make sure the backend is running before using the frontend.
You can replace the model.h5 with your updated trained model if needed.
For better performance, it’s recommended to use Chrome or Firefox to open the HTML file.

Technologies Used

Python: ML model training and backend server
TensorFlow/Keras: Model training
FastAPI: Serving the ML model via API
HTML: Frontend interface for testing

Authors

U R S L Bandara
A N S Silva 
