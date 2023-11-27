# main.py
from rest_api import app

import uvicorn
def main():

    uvicorn.run(app, host="127.0.0.1", port=8000)


#main
if __name__ == "__main__":
    main()