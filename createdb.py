import sqlite3  
  
def create_db():
	try:
		con = sqlite3.connect("db1.db")  
		print("Database opened successfully")  
		  
		# con.execute("create table Results (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, audio_file TEXT UNIQUE NOT NULL, audio_data TEXT UNIQUE NOT NULL, prediction TEXT UNIQUE NOT NULL, truth_value TEXT)")  
		
		con.execute("create table results (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, audio_file TEXT NOT NULL, prediction TEXT NOT NULL, truth_value TEXT, email TEXT NOT NULL)")  

		print("Table created successfully")  
		  
		con.close()
	except:
		print('Error occured while creating database <<most likely table already exists>>')