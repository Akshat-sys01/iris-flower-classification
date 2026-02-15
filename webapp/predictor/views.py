from django.shortcuts import render
import joblib
import os

# Load model once when server starts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "iris_model.pkl")

model = joblib.load(MODEL_PATH)

# Create your views here.
def home(request):
    prediction = None

    if request.method == "POST":
        # Get form values
        sepal_length = float(request.POST.get("sepal_length"))
        sepal_width = float(request.POST.get("sepal_width"))
        petal_length = float(request.POST.get("petal_length"))
        petal_width = float(request.POST.get("petal_width"))

        # Create input list
        input_data = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        # Predict
        result = model.predict(input_data)
        prediction = result[0]

    return render(request, 'predictor/home.html', {
        'prediction': prediction
    })