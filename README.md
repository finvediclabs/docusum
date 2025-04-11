## Environment Setup
python3.10 -m venv docusum                  
source docusum/bin/activate
## Install all dependencies using following command
pip install -r requirements.txt

## Run the applicaiton using 
pythong api.py

## Access required API's 
  http://localhost:7654/api/upload
  
    curl --location 'http://localhost:7654/api/upload' \
    --form 'file=@"/path/to/file"'
    
  http://localhost:7654//api/embed
  
    curl --location --request GET 'http://localhost:7654/api/embed' \
    --header 'Content-Type: application/javascript' \
    --data '{
    }'
    
  http://localhost:7654//api/question

    curl --location 'http://localhost:7654/api/question' \
    --header 'Content-Type: application/json' \
    --data '{
    "question": "List out OWASP top 10 ",
    "user_id": "docusum"
    }'
