from flask import Flask, render_template, request

from src.pipe.prediction_pipeline import CustomData, PredictionPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        sqft_living = request.form.get('sqft_living')
        sqft_lot = request.form.get('sqft_lot')
        floors = request.form.get('floors')
        waterfront = request.form.get('waterfront')
        view = request.form.get('view')
        condition = request.form.get('condition')
        grade = request.form.get('grade')
        sqft_above = request.form.get('sqft_above')
        sqft_basement = request.form.get('sqft_basement')
        yr_built = request.form.get('yr_built')
        yr_renovated = request.form.get('yr_renovated')
        total_rooms = request.form.get('total_rooms')

        custom_data = CustomData(sqft_living,sqft_lot,floors,waterfront,view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,total_rooms)
        input_qurie = custom_data.conver_data_to_dataframe()

        prediction_pipe = PredictionPipeline()
        prediction = prediction_pipe.predict(input_qurie)
        print(prediction)

        return render_template('form.html', prediction=prediction)

    return render_template('form.html')

if __name__ == "__main__":
    app.run(debug=True, port=8080, host='0.0.0.0')