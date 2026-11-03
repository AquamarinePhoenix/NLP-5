from nlplib.log import clear_log, write_log
from nlplib.data import read_raw_data

def main():
    clear_log()
    data = read_raw_data()
    write_log(data)

if __name__ == "__main__":
    main()