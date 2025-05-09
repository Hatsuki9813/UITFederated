from flask import Flask, request, jsonify
import mysql.connector
import subprocess
import json
import bcrypt

app = Flask(__name__)
current_model_name = None
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="federateddb"
)
logged_in_users = []
print(mydb)
@app.route('/login', methods=['POST'])
def login():
    cursor = mydb.cursor()
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()
    if result and bcrypt.checkpw(password.encode(), result[0].encode()):
        logged_in_users.append(username)
        return jsonify({"status": "success", "message": "Login successful", "user_id": result[0] })
    else:
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401

@app.route('/signup', methods=['POST'])
def signup():
    cursor = mydb.cursor()
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed))
        result = cursor.fetchone()
        return jsonify({"status": "success", "message": "Signup successful"})
    except mysql.connector.IntegrityError:
        return jsonify({"status": "error", "message": "Username already exists"}), 409


@app.route("/set-current-model", methods=["POST"])
def set_current_model():
    global current_model_name
    data = request.get_json()
    model = data.get("model")
    if not model:
        return "Thiếu tên model", 400
    current_model_name = model
    return "Model đã được lưu", 200
@app.route("/get-current-model", methods=["GET"])
def get_current_model():
    if current_model_name:
        return jsonify({"model": current_model_name})
    else:
        return "Model chưa được chọn", 404
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
@app.route("/get-global-model", methods=['POST'])
def get_global_model():
    # Sử dụng request từ Flask để lấy JSON data
    data = request.get_json()
    requested_model = data.get("model")
    if not requested_model:
        return "Lỗi: Không có model nào được yêu cầu"
    try:
        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM serverinfo WHERE GlobalModel = %s", (requested_model,))
        result = cursor.fetchone()
        if result:
            model_info = {
                "AggregatedMetric": result[0],
                "GlobalAccuracy": result[1],
                "GlobalLoss": result[2]
            }
            return json.dumps(model_info)
        else:
            return "Lỗi: Không tìm thấy thông tin cho model yêu cầu"
    except Exception as e:
        return f"Lỗi: {str(e)}"
@app.route("/update-client-info", methods=['POST'])
def update_client_info():
    data = request.get_json()
    model = data.get("model")
    clientdata = data.get("dataset")
    clientaccuracy = data.get("local_accuracy")
    clientloss = data.get("local_loss")
    clientname = data.get("clientname")
    try:
        cursor = mydb.cursor()
        cursor.execute("UPDATE client SET ClientModel = %s ClientData = %s, ClientAccuracy = %s, ClientLoss = %s WHERE ClientName = %s", (model, clientdata, clientaccuracy, clientloss, clientname))
        cursor.close()
        return f"Cập nhật thành công client"
    except Exception as e:
        return f"Lỗi: {str(e)}"
if __name__ == "__main__":
    app.run(debug=True)
   