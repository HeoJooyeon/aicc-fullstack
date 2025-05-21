from app import app,db
from flask import request, jsonify
from models import Friend

# READ
@app.route("/api/friends", methods = ["GET"])
def get_friends():
    friends = Friend.query.all() # select * from Friend
    result = [friend.to_json() for friend in friends]
    return jsonify(result)

# CREATE    
@app.route("/api/friends", methods = ["POST"])
def create_friend():
    try:
        data = request.json # 사용자가 입력한 값
        
        # validation
        required_fields = ["name","role","description","gender"]
        for required_field in required_fields:            
            if required_field not in data or not data.get(required_field): # key 값이 없거나  value 값이 없을 때
                return jsonify({"error": f"Missing required field: {required_field}"}), 400
        
        name = data.get("name")
        role = data.get("role")
        description = data.get("description")
        gender = data.get("gender")
        
        if gender == "male":
            img_url = f"https://avatar.iran.liara.run/public/boy?username={name}"
        elif gender == "female":
            img_url = f"https://avatar.iran.liara.run/public/girl?username={name}"
        else:
            img_url = None
        
        new_friend = Friend(name=name, role=role, description=description, gender=gender, img_url=img_url) # Friend의 새로운 data 생성
        
        db.session.add(new_friend)
        db.session.commit() # 진행 상황 실행 insert into freind (column) values (values)
        
        return jsonify({"msg": "friend created success"})
         
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

# DELETE
@app.route("/api/friends/<int:input_id>", methods = ["DELETE"])
def delete_friend(input_id):
    try:
        friend = Friend.query.get(input_id) # SQLAlchemy에서는 get을 primary key로 불러오는 것이 가능
        if friend is None:
            return jsonify({"error": "Friend not found"}), 400
        db.session.delete(friend)
        db.session.commit()
        
        return jsonify({"msg":"friend deleted success"}), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
    
# UPDATE
@app.route("/api/friends/<int:input_id>", methods = ["PATCH"])
def update_friend(input_id):
    try:
        friend = Friend.query.get(input_id)
        if friend is None:
            return jsonify({"error": "Friend not found"}), 400
        
        data = request.json
        
        friend.name = data.get("name", friend.name)
        friend.role = data.get("role", friend.role)
        friend.description = data.get("description", friend.description)
        friend.gender = data.get("gender", friend.gender)
        
        db.session.commit()
        return jsonify(friend.to_json()),200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
