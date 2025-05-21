from flask import Flask
from flask_sqlalchemy import SQLAlchemy # SQLite db 사용 가능
from flask_cors import CORS # 외부 API 호출 허용
import os

app = Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///friends.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app) # 연결된 상태

import routes

with app.app_context():
    db.create_all() # models class 구조로 data 형성 요청
    
if __name__ == "__main__":
    app.run(debug=True)