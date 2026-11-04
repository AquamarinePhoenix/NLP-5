#%%
from nlplib.log import clear_log, write_log
from nlplib.data import read_raw_data
from nlplib.db import (
    get_connection,
    create_dimMetadata,
    drop_table,
    insert_metadata,
    find_multiple_emotions,
    create_dimMetadataGrouped,
    insert_metadata_grouped,
    write_metadata
)
#%%
def main():
    #%%
    clear_log()
    data = read_raw_data()
    write_log(data)

    conn, cursor = get_connection()
    drop_table(conn, cursor, "dimMetadata")
    create_dimMetadata(conn, cursor)
    insert_metadata(conn, cursor, data)
#%%
    rows = find_multiple_emotions(conn, cursor,)
    write_log(rows)

    drop_table(conn, cursor, "dimMetadataGrouped")
    create_dimMetadataGrouped(conn, cursor)
    insert_metadata_grouped(conn, cursor)
    write_metadata(conn,cursor)

    write_log("Process ended successfully.")
#%%
if __name__ == "__main__":
    main()