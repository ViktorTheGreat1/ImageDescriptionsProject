import requests
from config import API_Key

#Model endpoint on Hugging Face
MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
imagesWithCaptions = {}
#Prepare headers with your API key
headers = {"Authorization": f"Bearer {API_Key}"}

def caption_single_image(imageFolder):
    """
    Loads the local image file "text.jpg" and sends it to the Hugging Face Inference API for captioning
    """
    if imageFolder.strip() == "":
        imageFolder = "images"
    with open(imageFolder, "rb") as f:
        for image_source in imageFolder:
            #1. Load image bytes
            try:
                with open(image_source, "rb") as f:
                    image_bytes = f.read()
            except Exception as e:
                print(f"Could not load image from {image_source}.\nError:{e}")
                return 
    
        #2. Send request to the Hugging Face Inference API
        response = requests.post(API_URL, headers=headers, files={"file": image_bytes})


        # 3. Parse the JSON response

        try:

            result = response.json()

        except Exception as e:

            print(f"Invalid JSON response: {e}")

            return

        #Check for errors
        if isinstance(result, dict) and "error" in result:
            print(f"[Error] {result['error']}")
            return
        
        #4. Extract caption
        caption = result[0].get("generated_text", "No caption found.")
        imagesWithCaptions[image_source] = caption
def main():
    #Caption the hardcoded file
    path = input("Enter the path to the image folder: ")
    caption_single_image(path)
    for image in imagesWithCaptions:
        print(f"Image: {image} \nCaption: {imagesWithCaptions[image]}\n")

if __name__ == "__main__":
    main()