import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="eco_pack",
        user="postgres",
        password="root"
    )