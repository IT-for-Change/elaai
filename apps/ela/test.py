import requests

# URL for Frappe server
url = "http://elaexplore.localhost:8081/api/method/upload_file"

# Headers for authentication
headers = {
    # Replace with your API key or token
    "Authorization": "token b909eccbd9b04d1:e7515a47bbfad81"
}

# Data fields (additional metadata, if required)
data = {
    # "doctype": "File",  # not required
    # "filename": "audio.mp3", #not required
    "is_private": 0
}

# Binary file you want to upload
files = {
    # Adjust the file path and MIME type
    'file': ('audio.mp3', open('../files/submissions/audio.mp3', 'rb'), 'application/octet-stream')
}

# Make the POST request
response = requests.post(url, headers=headers, data=data, files=files)
print(response.json())

# Check the response
if response.status_code == 200:
    print("File uploaded successfully!")
else:
    print(f"Error: {response.status_code} - {response.text}")
