from nlplib.log import clear_log, write_log
from nlplib.data import read_raw_data
from nlplib.db import get_connection, create_dimMetadata, drop_dimMetadata, insert_metadata

def main():
    clear_log()
    data = read_raw_data()
    write_log(data)

    conn, cursor = get_connection()
    drop_dimMetadata(conn, cursor)
    create_dimMetadata(conn, cursor)
    insert_metadata(conn, cursor, data)

if __name__ == "__main__":
    main()