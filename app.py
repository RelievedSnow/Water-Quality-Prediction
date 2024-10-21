from flask import Flask, render_template, request, send_file, url_for
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from water_quality_prediction import y_pred, y_test, X, mean_sqr_error, r2_sqr_error
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template("index.html", href='static/images/home_banner.jpg', avatar='static/images/avatar.jpg')
    else:
        # Get form data from the request
        BOD = request.form['BOD']
        FC = request.form['FC']
        FS = request.form['FS']
        pH = request.form['pH']

        # Convert inputs to numpy array
        input_data = [BOD, FC, FS, pH]
        input_data = np.array(input_data, dtype=float).reshape(1, -1)

        # Load the model
        model = load('water_quality_model.joblib')

        # Make prediction
        prediction = model.predict(input_data)

        # Extract predicted value (assuming it's dissolved oxygen, DO)
        predicted_do = prediction[0]

        # Condition to classify the result based on predicted_do
        if predicted_do > 8:
            result = "Good"
        elif predicted_do >= 5:
            result = "Moderate"
        else:
            result = "Poor"

        mse = mean_sqr_error
        r2 = r2_sqr_error
        makePicture('RS_Session_259_AU_1203_1.csv', model, input_data, 'do_prediction_plot.png', 'do_prediction_bar_plot.png')

        # Render result template with the prediction and the plot URL
        scatter_plot_url = url_for('static', filename='images/do_prediction_scatter_plot.png')
        bar_plot_url = url_for('static', filename='images/do_prediction_bar_plot.png')
        return render_template("result.html", prediction=result, predicted_do=predicted_do, mse=mse, r2=r2, plot_url=scatter_plot_url, bar_plot_url=bar_plot_url,)


def makePicture(traning_data_file, model, input_data, outputfile_1, outputfile_2):
    # Visualize actual vs predicted values for DO (mg/L)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='blue', label='Predicted DO')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label='Perfect Prediction Line')
    plt.xlabel('Actual DO (mg/L)')
    plt.ylabel('Predicted DO (mg/L)')
    plt.title('Actual vs Predicted Dissolved Oxygen (DO) Levels')
    plt.legend()
    plt.grid(True)
    # Save the plot as an image in the static folder
    plot_path = os.path.join('static', 'images', 'do_prediction_scatter_plot.png')
    plt.savefig(plot_path)
    plt.close()


    # Feature importance visualization
    feature_importances = model.feature_importances_
    features = X.columns

    plt.figure(figsize=(8, 6))
    sns.barplot(x=feature_importances, y=features, color="blue")
    plt.title('Feature Importance for DO Prediction')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plot_path = os.path.join('static', 'images', 'do_prediction_bar_plot.png')
    plt.savefig(plot_path)
    plt.close()

if __name__ == '__main__':
    app.run(debug=True)
