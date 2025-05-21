from flask import Flask, render_template, request
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

def get_db_connection():
    return pymysql. connect(
        host = "localhost",
        user = "root",
        password = "1111",
        db = "bookdb",
        charset = "utf8mb4"
    )

@app.route("/", methods=["GET", "POST"]) # get, post http://localhost:5000/
def index():
    if request.method == "POST":
        keyword = request. form["keyword"]

        from fetch_books import fetch_books_to_mysql
        fetch_books_to_mysql(keyword)
        
        conn = get_db_connection()
        
        query = f"""
            SELECT title, pageCount FROM bookdb.books
            WHERE search_keyword = %s AND pageCount > 0
            ORDER BY pageCount DESC LIMIT 10
        """
        df = pd.read_sql(query, conn, params=(keyword,))
        conn.close()
        


        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        
        if not df.empty:
            plt.figure(figsize=(10, 6))
            plt.barh(df["title"], df["pageCount"])
            plt.xlabel("Page Count")
            plt.title(f"Top 10 Longest Books for '{keyword}'")
            plt.tight_layout ()
            
            if not os.path.exists("static"):
                os.makedirs("static")
                
            plt.savefig("static/chart.png")
            plt.close()
            
            return render_template("index.html", chart_exists=True, keyword=keyword)
            
        return render_template("index.html", message="데이터가 없습니다.")
    
    return render_template("index.html", chart_exists=False)


if __name__ == "__main__":
    app.run(debug=True)