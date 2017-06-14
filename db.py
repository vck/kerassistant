import sqlite3 as sqlite

db = sqlite.connect('learning.db')
con.execute("CREATE TABLE experience(id INTEGER PRIMARY KEY AUTOINCREMENT, model_id text, score INT)")

