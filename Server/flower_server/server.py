from flask import Flask
import mysql.connector
import subprocess
import json
app = Flask(__name__)

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="federateddb"
)

print(mydb)

def startsuperlink():
    command = ["flower-superlink", "--insecure"]
    subprocess.Popen(command)
startsuperlink()
@app.route("/check-db")
def check_db():
    try:
        cursor = mydb.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        cursor.close()
        return f"Kết nối thành công! MySQL version: {version[0]}"
    except mysql.connector.Error as err:
        return f"Kết nối thất bại: {err}"
@app.route("/updateparameters")
def updateparameters():
    try:
        with open("final_accuracy.json", "r") as file:
            data = json.load(file)
            global_accuracy = data.get("GlobalAccuracy", None)
        if global_accuracy is None:
            return "Lỗi: Không tìm thấy GlobalAccuracy trong file JSON"
        with open("aggregated_loss.json", "r") as file2:
            data = json.load(file2)
            aggregated_loss = data.get("aggregated_loss", None)
        if aggregated_loss is None:
            return "Lỗi: Không tìm thấy aggregated_loss trong file json"
        cursor = mydb.cursor()
        cursor.execute("UPDATE serverinfo SET GlobalAccuracy = %s", (global_accuracy,))
        cursor.execute("update serverinfo SET GlobalLoss = %s", (aggregated_loss,))
        mydb.commit()
        cursor.close()
    except Exception as e:
        return f"Lỗi: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
