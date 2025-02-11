import os
import base64
import openai
from PIL import Image

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def resize_image(image_path, max_low_res = 512):
    # High Resolution parameters in OpenAI
    # Short side: less than 768px
    # Long side: less than 2,000px
    
    img = Image.open(image_path)
    width, height = img.size
    resize_factor_width = width/max_low_res
    resize_factor_height = height/max_low_res

    if resize_factor_width>1.0 or resize_factor_height>1.0:
        if resize_factor_height >= resize_factor_width:
            resize_factor = resize_factor_height
        else:
            resize_factor = resize_factor_width

        resized_image_path = image_path.split(".")[0] + "_resize.jpg"
        if not os.path.isfile(resized_image_path):
            new_width = width/resize_factor
            new_height = height/resize_factor
            resized_img = img.resize((int(new_width), int(new_height)))
            resized_img.save(resized_image_path)
        final_image_path = resized_image_path
    else:
        final_image_path = image_path
    return final_image_path

def prompt_openai(prompt, images):
    img_detail = "low"
    resolution = 512

    content = [{ "type": "text",
                "text": prompt,
                }]
    for image_path in images:
        resized_image_path = resize_image(image_path, max_low_res=resolution)
        base64_image = encode_image(resized_image_path)
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}",
                                      "detail": img_detail
                                      },
        })

    client = openai.OpenAI(api_key = os.environ["PERSONAL_OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        max_tokens=300,
    )
    print(response.choices[0])

def prompt_deepsek():
    pass

def prompt_claude():
    pass

def prompt_claude():
    pass

if __name__ == "__main__":
    images = ["eu_hri_dataset/sequences/001/1496771406_841137459.jpg",
              "eu_hri_dataset/sequences/001/1496771407_400287081.jpg",
              "eu_hri_dataset/sequences/001/1496771408_141694645.jpg"]
    prompt = "What are in these images? Is there any difference between them?"
    prompt_openai(prompt, images)