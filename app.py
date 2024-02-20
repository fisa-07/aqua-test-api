import pandas as pd
from flask import Flask,jsonify;
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

#Loading the dataset
df = pd.read_csv('D:\\Python Code\\train1.csv')

x_train, x_test, y_train, y_test = train_test_split(df.drop('CLASS', axis=1), df['CLASS'], test_size=0.2, random_state=42)

# Create a random forest classifier
rfc = RandomForestClassifier()

# Training the model using the training data
rfc.fit(x_train, y_train)

@app.route('/')
def ml_model():
    test_data = [[3044, 15856, 20114, -5594, 6915, -2732, 15199, -2292, 14491, -2392, 16643, -1362]]
          
    rfc_pred = rfc.predict(test_data)[0] # rfc.predict() gives data in a nested list. So selecting list at 0th index
    dict = {
        0: "Acetic Acid",
        1: "Acetone",
        2: "Ammonia",
        3: "Ethanol",
        4: "Formic Acid",
        5: "Hydrochloric Acid",
        6: "Hydrogen Peroxide",
        7: "Phosphoric Acid",
        8: "Sodium Hypochlorite",
        9: "Sulphuric Acid",
        10: "Waste water"
    }
    class_name = {"result":f"{dict[rfc_pred]}"}
    return jsonify(class_name)

# if __name__ == '__main__':
#     app.run(debug=True)